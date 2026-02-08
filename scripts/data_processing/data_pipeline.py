import tensorflow as tf
from config.settings import DATA_CONFIG

class DataPipeline:
    """Simple, efficient data loader."""
    
    def __init__(self, img_size=(224, 224), batch_size=32):
        self.img_size = img_size
        self.batch_size = batch_size
    
    def get_datasets(self):
        """Load train/val/test datasets."""
        
        # Training data with augmentation
        train_ds = tf.keras.utils.image_dataset_from_directory(
            str(DATA_CONFIG.TRAIN),
            seed=42,
            image_size=self.img_size,
            batch_size=self.batch_size,
            label_mode="categorical",
            shuffle=True
        )
        
        # Add augmentation
        augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ])
        
        train_ds = train_ds.map(
            lambda x, y: (
                tf.keras.applications.mobilenet_v2.preprocess_input(
                    augmentation(x, training=True)
                ),
                y
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(tf.data.AUTOTUNE)
        
        # Validation data (no augmentation)
        val_ds = tf.keras.utils.image_dataset_from_directory(
            str(DATA_CONFIG.VAL),
            seed=42,
            image_size=self.img_size,
            batch_size=self.batch_size,
            label_mode="categorical",
            shuffle=False
        )
        
        val_ds = val_ds.map(
            lambda x, y: (
                tf.keras.applications.mobilenet_v2.preprocess_input(x),
                y
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(tf.data.AUTOTUNE)
        
        # Test data
        test_ds = tf.keras.utils.image_dataset_from_directory(
            str(DATA_CONFIG.TEST),
            seed=42,
            image_size=self.img_size,
            batch_size=self.batch_size,
            label_mode="categorical",
            shuffle=False
        )
        
        test_ds = test_ds.map(
            lambda x, y: (
                tf.keras.applications.mobilenet_v2.preprocess_input(x),
                y
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(tf.data.AUTOTUNE)
        
        return train_ds, val_ds, test_ds

if __name__ == "__main__":
    print("🔧 Testing DataPipeline...\n")

    pipeline = DataPipeline()
    train_ds, val_ds, test_ds = pipeline.get_datasets()

    print(f"✅ Train dataset: {len(train_ds)} batches")
    print(f"✅ Val dataset: {len(val_ds)} batches")
    print(f"✅ Test dataset: {len(test_ds)} batches")

    print("\n✅ DataPipeline loaded successfully!")