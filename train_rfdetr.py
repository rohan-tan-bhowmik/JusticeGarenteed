from rfdetr import RFDETRBase
import argparse

def main():

    parser = argparse.ArgumentParser(description="RF-DETR Training Configuration")

    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='COCO-formatted dataset directory with train, valid, and test folders, each containing _annotations.coco.json')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save training artifacts like checkpoints and logs')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size per iteration')
    parser.add_argument('--grad_accum_steps', type=int, default=1,
                        help='Number of gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for the full model')
    parser.add_argument('--lr_encoder', type=float, default=1e-5,
                        help='Learning rate for the encoder backbone')
    parser.add_argument('--resolution', type=int, default=(1792, 1008),
                        help='Input image resolution, must be divisible by 56')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='L2 regularization weight decay')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to train on (cuda or cpu)')
    parser.add_argument('--use_ema', action='store_true',
                        help='Enable Exponential Moving Average of weights')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                        help='Enable gradient checkpointing to reduce memory usage')
    parser.add_argument('--checkpoint_interval', type=int, default=1,
                        help='Epoch interval to save model checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training')
    parser.add_argument('--tensorboard', action='store_true',
                        help='Enable TensorBoard logging')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--project', type=str, default='rf-detr',
                        help='WandB project name')
    parser.add_argument('--run', type=str, default='default_run',
                        help='WandB run name')

    # Early Stopping
    parser.add_argument('--early_stopping', action='store_true',
                        help='Enable early stopping based on mAP')
    parser.add_argument('--early_stopping_patience', type=int, default=5,
                        help='Number of epochs without mAP improvement to wait before stopping')
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.0,
                        help='Minimum mAP improvement to reset early stopping counter')
    parser.add_argument('--early_stopping_use_ema', action='store_true',
                        help='Use EMA model weights for early stopping checks')

    model = RFDETRBase()

    args = parser.parse_args()
    model.train(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        lr_encoder=args.lr_encoder,
        resolution=args.resolution,
        weight_decay=args.weight_decay,
        device=args.device,
        use_ema=args.use_ema,
        gradient_checkpointing=args.gradient_checkpointing,
        checkpoint_interval=args.checkpoint_interval,
        resume=args.resume,
        tensorboard=args.tensorboard,
        wandb=args.wandb,
        project=args.project,
        run=args.run,
        early_stopping=args.early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        early_stopping_use_ema=args.early_stopping_use_ema
    )