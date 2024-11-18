import os
import gc
import sys
import json
import time
import torch
import random
import signal
import argparse
import numpy as np
import torchvision
import pandas as pd
from PIL import Image
from glob import glob
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split

from mean_ap import MeanAveragePrecision

# python faster_rcnn.py -gpu 2 -in ./Dataset -ep 350 -bs 64 -pre True -ds 0.7 0.2 0.1 -save 50
# python faster_rcnn.py -gpu 3 -in ./Dataset -ep 350 -bs 64 -pre False -ds 0.7 0.2 0.1 -save 50

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(69)


def cleanup(signum, frame):
    """Catching interrupts and cleaning up cache"""
    print("\n\n\t<Interrupt Received. Cleaning Up...>")
    
    torch.cuda.empty_cache()
    sys.exit(0)


class CustomDataset(Dataset):
    """Custom Dataset class for loading data"""
    
    def __init__(self, dataset_dir, transforms=None, split='train', dataset_split={"train":0.7, "test":0.2, "val":0.1}):
        """Fn. to load dataset from memory"""
        self.transforms = transforms
        all_files = glob(os.path.join(dataset_dir, '**', '*.json'), recursive=True)
        image_paths = []
        annotations = []
        
        # [Check dataset split values]
        if not np.isclose(sum(dataset_split.values()), 1.0):
            print("\n\n<Improper dataset split values! Shifting to 70-20-10 split>\n")
            dataset_split = {"train":0.7, "test":0.2, "val":0.1}

        # [Get annotations from json files]
        for filepath in all_files:
            img_path = f"{os.path.splitext(filepath)[0]}.png"
            if os.path.exists(img_path):
                image_paths.append(img_path)
                with open(filepath, 'r') as f_json:
                    annotations.append(json.load(f_json))

        # [Dataset Split]
        # First we split dataset into training dataset and other
        train_images, temp_images, train_annotations, temp_annotations = train_test_split(
            image_paths, annotations, train_size=dataset_split["train"], random_state=42)
        # Val set = 15% of original dataset => Validation set = 15/(1-train)% of remaining dataset
        val_fraction = dataset_split["val"]/(1 - dataset_split["train"])
        # Next we split remaining data into validation and test
        val_images, test_images, val_annotations, test_annotations = train_test_split(
            temp_images, temp_annotations, train_size=val_fraction, random_state=42)

        if split == 'train':
            self.image_paths, self.annotations = train_images, train_annotations
        elif split == 'val':
            self.image_paths, self.annotations = val_images, val_annotations
        elif split == 'test':
            self.image_paths, self.annotations = test_images, test_annotations

    def __len__(self):
        """Override the len fn. to return length of dataset"""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Override getitem fn. to return single entry from dataset"""
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        target = self.annotations[idx]
        
        class_labels = torch.tensor([int(ann['class']) for ann in target['labels']], dtype=torch.int64)
        boxes = torch.tensor([[ann['x'], ann['y'], ann['x']+ann['width'], ann['y']+ann['height']] 
                              for ann in target['labels']], dtype=torch.float32)
        
        target = {
            "boxes": boxes,
            "labels": class_labels,
            "image_id": torch.tensor([idx]),
            "iscrowd": torch.zeros(len(class_labels), dtype=torch.int64),
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        }
        
        if self.transforms:
            img = self.transforms(img)
        return img, target, img_path
    

def create_graph(train_losses, test_losses, model_mAPs, classes_mAPs, epoch_num, loss_dict, store_location, graph_title):
    """Creates a plot of losses and mAP values and stores in mentioned location"""
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 20))
    fig.suptitle(graph_title)

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Plot each loss type
    for idx, (loss_name, loss_title) in enumerate(loss_dict.items()):
            # Plot training loss
            axes[idx].plot(range(1, epoch_num + 1), train_losses[loss_name], label='Train', color='b', marker='o')
            # Plot test loss
            axes[idx].plot(range(1, epoch_num + 1), test_losses[loss_name], label='Test', color='r', marker='x')
            # Add titles and labels
            axes[idx].set_title(loss_title)
            axes[idx].set_xlabel("Epoch")
            axes[idx].set_ylabel("Loss")
            axes[idx].legend()
            
    # Code to plot mAP
    idx = len(loss_dict.keys())
    axes[idx].plot(range(1, epoch_num + 1), model_mAPs, label='Overall mAP', color='r', marker='x')
    for class_num, class_mAP_change in enumerate(classes_mAPs):
        axes[idx].plot(range(1, epoch_num + 1), class_mAP_change, label=f'Class {class_num + 1} mAP', marker='o')
    axes[idx].set_title("Mean Average Precision (mAP)")
    axes[idx].set_xlabel("Epoch")
    axes[idx].set_ylabel("mAP")
    axes[idx].legend()
            
    # Save the figure with tight bounding box
    plt.savefig(store_location)
    plt.show()
    plt.close()


def visualize_random_images_with_boxes(predictions, output_folder, epoch_num, image_paths, iou_threshold=0.5):
    """Visualizes three random samples from the predictions with bounding boxes, side by side."""
    
    if len(image_paths) < 3:
        raise ValueError(f"Not enough images in image_paths to sample from. Found {len(image_paths)}, but need 3.")

    # Initialize a list to hold the images for display
    images_with_bboxes = []
    
    # Select three random indices
    random_indices = random.sample(range(len(image_paths)), 3)
    
    # Loop over the three random indices
    for idx in random_indices:
        img_path = image_paths[idx]
        img = torchvision.io.image.read_image(img_path)

        # Extract boxes, labels, and scores for the selected image
        boxes = predictions[idx]["boxes"]
        labels = predictions[idx]["labels"]
        scores = predictions[idx]["scores"]

        # Perform NMS (Non-Maximum Suppression) to remove overlapping boxes
        keep_indices = torchvision.ops.nms(boxes, scores, iou_threshold)

        # Filter boxes, labels, and scores using NMS results
        boxes = boxes[keep_indices]
        labels = [labels[i].item() for i in keep_indices]
        labels = [str(label) for label in labels]  # Convert labels to strings

        # Draw bounding boxes on the image
        img_with_bboxes = torchvision.utils.draw_bounding_boxes(img, boxes=boxes, labels=labels, colors="red", width=2)

        # Convert the image from tensor to numpy array for display
        img_with_bboxes_np = img_with_bboxes.detach().cpu().numpy()
        img_with_bboxes_np = img_with_bboxes_np.transpose(1, 2, 0)  # Convert from CHW to HWC format
        
        # Append the processed image to the list
        images_with_bboxes.append(img_with_bboxes_np)
        
        del img_with_bboxes_np, boxes, labels, scores
        torch.cuda.empty_cache()
        gc.collect()

    # Create a single image with three subplots (side by side)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Display each image in the respective subplot
    for i, ax in enumerate(axes):
        ax.imshow(images_with_bboxes[i])
        ax.axis('off')  # Turn off axes for better visualization

    # Save the figure with three images side by side
    plt.tight_layout()
    plt.savefig(f'./{output_folder}/preds/detected_at_epoch-{epoch_num}.png')
    plt.close()  # Close the figure after saving to avoid excessive memory usage
    
    del images_with_bboxes
    torch.cuda.empty_cache()
    gc.collect()


def calculate_mean_average_precision(predictions, targets, epoch_num, output_folder_name):
    # Initialize MeanAveragePrecision with class metrics enabled
    metric = MeanAveragePrecision(class_metrics=True)

    # Update metric with predictions and ground-truth targets
    metric.update(predictions, targets)

    # Compute the mAP metrics
    map_results = metric.compute()

    full_map_scores = {
        "map": map_results.map.item(),
        "map_50": map_results.map_50.item(),
        "map_75": map_results.map_75.item(),
        "map_small": map_results.map_small.item(),
        "map_medium": map_results.map_medium.item(),
        "map_large": map_results.map_large.item(),
        "mar_1": map_results.mar_1.item(),
        "mar_10": map_results.mar_10.item(),
        "mar_100": map_results.mar_100.item(),
        "mar_small": map_results.mar_small.item(),
        "mar_medium": map_results.mar_medium.item(),
        "mar_large": map_results.mar_large.item(),
        "map_per_class": map_results.map_per_class.tolist(),
        "mar_100_per_class": map_results["mar_100_per_class"].tolist(),
        "classes": map_results.classes.tolist(),
    }
    
    return full_map_scores

    
def validate_model(model, device, loader, output_folder, epoch_num, num_classes):
    """Runs GPU-accelerated model evaluation, applies NMS, and calculates mAP for all classes and IoU thresholds."""
    model.to(device)
    model.eval()
    predictions = []
    targets = []
    image_paths = []
    
    validation_start_time = time.time()

    with torch.no_grad():
        for images, targs, img_paths in tqdm(loader, desc=f"Validation Epoch {epoch_num}", unit="batch", total=len(loader)):
            images = [img.to(device) for img in images]
            preds = model(images)
            # print(preds)            
            # print(targs)
            
            for i, pred in enumerate(preds):
                pred["boxes"] = pred["boxes"].to(device)
                pred["scores"] = pred["scores"].to(device)
                pred["labels"] = pred["labels"].to(device)
                targs[i]["boxes"] = targs[i]["boxes"].to(device)
                targs[i]["labels"] = targs[i]["labels"].to(device)

                # Apply NMS for each prediction
                nms_indices = torchvision.ops.nms(pred["boxes"], pred["scores"], iou_threshold=0.5)
                pred["boxes"] = pred["boxes"][nms_indices]
                pred["scores"] = pred["scores"][nms_indices]
                pred["labels"] = pred["labels"][nms_indices]
                
            predictions.extend(preds)
            targets.extend(targs)
            image_paths.extend(img_paths)
            
            del images, preds
            torch.cuda.empty_cache()
            gc.collect()
    
    # Calculate mAP for each class and for the entire model
    full_mAP_scores = calculate_mean_average_precision(predictions, targets, epoch_num=epoch_num, output_folder_name=output_folder)
    
    print(f"\nValidation (Time Taken - {(time.time() - validation_start_time):.4f}s):\n"
          f"\tPer Class -> {[f'{x:.4f}' for x in full_mAP_scores['map_per_class']]}"
          f"\n\tOverall mAP -> {full_mAP_scores['map']:.4f} or {np.mean(full_mAP_scores['map_per_class']):.4f}")

    visualize_random_images_with_boxes(predictions, output_folder, epoch_num, image_paths)
    torch.cuda.empty_cache()
    gc.collect()
    
    return full_mAP_scores


def save_checkpoint(model, optimizer, epoch, filename='checkpoint.pth'):
    """Code to save model checkpoints"""
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(checkpoint, filename)
    print(f"<Checkpoint saved to {filename}>")
    
    
def save_to_csv(filename, epoch_number, training_loss, testing_loss, full_mAP_scores, loss_desc):
    """Save loss values and evaluation metrics to csv"""
    
    # Store all data into a dictionary
    epoch_data = {}
    epoch_data["Epoch"] = epoch_number
    
    for key, value in loss_desc.items():
        epoch_data[f"{value} Training"] = training_loss[key]
        epoch_data[f"{value} Testing"] = testing_loss[key]
        
    for key, value in full_mAP_scores.items():
        epoch_data[key] = value        
        
    # Convert dictionary to dataframe and append to pre-existing data (if any)
    if os.path.exists(filename):
        existing_df = pd.read_csv(filename)
        updated_df = pd.concat([existing_df, pd.DataFrame([epoch_data])], ignore_index=True)
    else:
        updated_df = pd.DataFrame([epoch_data])
    updated_df.to_csv(filename, index=False)
    print("\n<Loss data written to CSV>")
    

def collate_fn(batch):
    """Collate function for DataLoader"""
    return tuple(zip(*batch))


def test_model(model, device, loader, epoch):
    """Testing Function with GradScaler for mixed precision"""
    
    model.to(device)
    model.train()
    dict_aggregate = {'loss_classifier':0, 'loss_box_reg':0, 'loss_objectness':0, 'loss_rpn_box_reg':0, 'total':0}
    epoch_start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (images, targets, image_paths) in tqdm(enumerate(loader), desc=f"Testing Epoch {epoch}", unit="batch", total=len(loader)):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with autocast():  # Mixed precision for testing
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            
            # Store loss for batch
            for key, value in loss_dict.items():
                dict_aggregate[key] += value.item()
            dict_aggregate['total'] += losses.item()

            del images, targets, loss_dict
            torch.cuda.empty_cache()
            gc.collect()
                
    for key in dict_aggregate.keys():
        dict_aggregate[key] /= len(loader)
    print(f"\t\tTest Epoch: {epoch}\tAverage Loss: {dict_aggregate['total']:.6f}\tTime: {(time.time() - epoch_start_time):.4f}s\n")
    return dict_aggregate


def train_model(model, device, loader, optimizer, epoch, scaler):
    """Training Function with GradScaler for mixed precision"""
    
    model.to(device)
    model.train()
    dict_aggregate = {'loss_classifier':0, 'loss_box_reg':0, 'loss_objectness':0, 'loss_rpn_box_reg':0, 'total':0}
    epoch_start_time = time.time()
    
    # Running through batches in dataset loader
    for batch_idx, (images, targets, image_paths) in tqdm(enumerate(loader), desc=f"Training Epoch {epoch}", unit="batch", total=len(loader)):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        with autocast():  # Mixed precision for training
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        scaler.scale(losses).backward()  # Scale loss for mixed-precision training
        scaler.step(optimizer)           # Step with scaled optimizer
        scaler.update()                  # Update scaler
            
        # Store loss for batch
        for key, value in loss_dict.items():
            dict_aggregate[key] += value.item()
        dict_aggregate['total'] += losses.item()
        
        del images, targets, loss_dict, losses
        torch.cuda.empty_cache()
        gc.collect()
        

    for key in dict_aggregate.keys():
        dict_aggregate[key] /= len(loader)
    print(f"\t\tTrain Epoch: {epoch}\tAverage Loss: {dict_aggregate['total']:.6f}\tTime: {(time.time() - epoch_start_time):.4f}s\n")
    return dict_aggregate


def main(directory, num_epochs, batch_size=4, pretrain=True, learning_rate = 0.005, momentum = 0.9, weight_decay=0.005, data_split={"train":0.6, "test":0.2, "val":0.2}, save_after=10, device='cuda'):
    """Main Training Loop"""
    
    # [Create necessary folders]
    output_folder_name = f"Ep-{num_epochs}_BS-{batch_size}_Pre-{pretrain}_LR-{learning_rate}_Mo-{momentum}_WD-{weight_decay}_Day-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(output_folder_name, exist_ok=True)
    os.makedirs(os.path.join(output_folder_name, "chkpts"), exist_ok=True)
    os.makedirs(os.path.join(output_folder_name, "preds"), exist_ok=True)
    
    # Get number of classes based on subfolders in directory
    num_classes = len([entry for entry in os.listdir(directory) if os.path.isdir(os.path.join(directory, entry))])
    
    # [Get Dataset Split from Custom Dataset]
    transform_to_tensor = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    train_dataset = CustomDataset(directory, transforms=transform_to_tensor, split='train', dataset_split=data_split)
    test_dataset = CustomDataset(directory, transforms=transform_to_tensor, split='test', dataset_split=data_split)
    val_dataset = CustomDataset(directory, transforms=transform_to_tensor, split='val', dataset_split=data_split)

    # [Define DataLoaders]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    print("<Dataset and DataLoaders defined>")

    # [Load fasterrcnn_resnet50_fpn_v2 from PyTorch library]
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(pretrained=pretrain)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes + 1)  # Adjust for number of classes
    print("<Faster RCNN ResNet50 FPN v2 Model Loaded from PyTorch Library>")

    # [Define Optimiser and GradScaler for using mixed precision]
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    scaler = GradScaler()
    
    # Create descriptions for different losses
    loss_desc = {"loss_classifier":"Classification Loss (Fast RCNN)", "loss_box_reg":"Box Regression Loss (Fast RCNN)", "loss_objectness":"Objectness Loss (RPN)", "loss_rpn_box_reg":" Box Regression Loss (RPN)", "total":"Total Loss"}
    
    # [Initialise dictionary with lists to store different losses]
    test_losses = {key : [] for key in loss_desc.keys()}
    train_losses = {key : [] for key in loss_desc.keys()}
    
    model_mAP_change = []
    classes_mAP_change = [[] for _ in range(num_classes)]

    # [Training the model on the dataset]
    try:
        for epoch in range(1, num_epochs + 1):
            print(f"\n\t\t\tEpoch: [{epoch}/{num_epochs}]")
            train_loss_epoch = train_model(model, device, train_loader, optimizer, epoch, scaler)
            test_loss_epoch = test_model(model, device, test_loader, epoch)
            full_mAP_epoch = validate_model(model, device, val_loader, output_folder_name, epoch, num_classes)  # Validation step with mAP

            # Append losses and model_mAP to respective lists
            for loss_name in loss_desc.keys():
                test_losses[loss_name].append(test_loss_epoch[loss_name])
                train_losses[loss_name].append(train_loss_epoch[loss_name])

            model_mAP_change.append(full_mAP_epoch['map'])
            for cNum, class_mAP_value in enumerate(full_mAP_epoch['map_per_class']):
                classes_mAP_change[cNum].append(class_mAP_value)
            
            save_to_csv(os.path.join(output_folder_name, "training_stats.csv"), epoch, train_loss_epoch, test_loss_epoch, full_mAP_epoch, loss_desc)
            if epoch % save_after == 0 or epoch == num_epochs:
                save_checkpoint(model, optimizer, epoch, os.path.join(output_folder_name, "chkpts", f"epoch_{epoch}.pth"))
                
            del train_loss_epoch, test_loss_epoch, full_mAP_epoch
            torch.cuda.empty_cache()
            gc.collect()
    except KeyboardInterrupt:
        print("\n\n\t<Interrupt Received. Cleaning Up...>")
        
        torch.cuda.empty_cache()
        sys.exit(0)
        

    # Create a plot of the losses and metrics
    create_graph(
        train_losses = train_losses,
        test_losses = test_losses,
        model_mAPs = model_mAP_change,
        classes_mAPs = classes_mAP_change,
        epoch_num = num_epochs,
        loss_dict = loss_desc,
        store_location = os.path.join(output_folder_name, "metrics_plotted.png"),
        graph_title = f"Faster RCNN ResNET FPN v2\nEp-{num_epochs}_BS-{batch_size}_Pre-{pretrain}_LR-{learning_rate}_Mo-{momentum}_WD-{weight_decay}"
    )
    
    del model, optimizer, scaler
    del train_losses, test_losses, model_mAP_change, classes_mAP_change
    del train_dataset, train_loader, test_dataset, test_loader, val_dataset, val_loader
    torch.cuda.empty_cache()
    gc.collect()

    
    
if __name__ == "__main__":
    """First Fn. to be called when program is run"""
    
    # Call function to clear cache when interrupted
    signal.signal(signal.SIGINT, cleanup)   # By Ctrl + C
    signal.signal(signal.SIGTSTP, cleanup)  # By Ctrl + Z
    
    # Initialize Argument Parser
    parser = argparse.ArgumentParser(description="Faster R-CNN for Object Detection")
    parser.add_argument('-gpu', '--gpu_id', type=int, required=True, help='ID of GPU')
    parser.add_argument('-in', '--dataset', type=str, required=True, help='Path to dataset folder')
    parser.add_argument('-ep', '--epoch_number', type=int, required=True, help='Number of epochs')
    parser.add_argument('-bs', '--batch_size', type=int, required=True, help='Batch size')
    parser.add_argument('-pre', '--pretrain', type=int, required=True, help='Use pretrained model (0 - False, 1 - True)')
    parser.add_argument('-ds', '--data_split', type=float, nargs='+', help='Train/test/val split')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.005, help='Learning rate')
    parser.add_argument('-mo', '--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0005, help='Weight decay')
    parser.add_argument('-save', '--save_after', type=int, default=10, help='Checkpoint save frequency')
    args = parser.parse_args()

    args.pretrain = False if args.pretrain == 0 else True
    main(
        directory      =   args.dataset, 
        num_epochs     =   args.epoch_number, 
        batch_size     =   args.batch_size, 
        pretrain       =   args.pretrain, 
        learning_rate  =   args.learning_rate, 
        momentum       =   args.momentum,
        weight_decay   =   args.weight_decay, 
        data_split     =   {"train":args.data_split[0], "test":args.data_split[1], "val":args.data_split[2]},
        save_after     =   args.save_after,
        device         =   f'cuda:{args.gpu_id}'
    )
    
    torch.cuda.empty_cache()
