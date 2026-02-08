# train_data_prep.py
import os
import requests
import zipfile
from pathlib import Path
import shutil
import json

class HousePlantDatasetBuilder:
    """
    Build curated house plant disease dataset from:
    - PlantVillage (filtered)
    - iNaturalist API (indoor observations)
    - Custom annotations
    """
    
    def __init__(self, data_dir="./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.plant_dir = self.data_dir / "house_plants"
        self.plant_dir.mkdir(exist_ok=True)
        
    def download_plantvillage(self):
        """Download PlantVillage from Kaggle (requires kaggle API key)"""
        print("📥 Downloading PlantVillage dataset...")
        
        # Requires: kaggle.json in ~/.kaggle/
        os.system("kaggle datasets download -d emmarex/plantdisease -p ./data/raw --unzip")
        
        # Filter for house plants only
        pv_path = Path("./data/raw/PlantVillage")
        house_plants = [
            "Tomato", "Pepper", "Potato", "Cucumber", "Bean",
            "Apple", "Grape", "Orange", "Peach",
        ]
        
        for plant in house_plants:
            for ext in [plant, plant.lower()]:
                src = pv_path / ext
                if src.exists():
                    dst = self.plant_dir / ext
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
                    print(f"✓ Copied {ext}")
                    
    def download_inaturalist_houseplants(self):
        """
        Download iNaturalist observations for indoor/houseplants
        API: https://api.inaturalist.org/v1/observations
        """
        print("📥 Downloading iNaturalist house plant observations...")
        
        # Common houseplant species (iNat taxon IDs)
        plants_to_download = {
            "Monstera": 143624,
            "Pothos": 76348,
            "Snake_Plant": 97234,
            "Philodendron": 47126,
            "Anthurium": 48294,
            "Orchid": 51856,
            "Ficus": 48397,
        }
        
        for plant_name, taxon_id in plants_to_download.items():
            os.makedirs(self.plant_dir / plant_name / "healthy", exist_ok=True)
            
            # Query iNat API
            url = f"https://api.inaturalist.org/v1/observations"
            params = {
                "taxon_id": taxon_id,
                "photos": True,
                "captive_cultivated": True,  # Only indoor/cultivated
                "quality_grade": "research",
                "per_page": 200,
                "order_by": "created_at",
                "order": "desc"
            }
            
            response = requests.get(url, params=params)
            obs = response.json()["results"]
            
            count = 0
            for observation in obs:
                if observation.get("photos"):
                    for photo in observation["photos"]:
                        try:
                            img_url = photo["url"].replace("square", "medium")
                            img_data = requests.get(img_url).content
                            
                            filename = self.plant_dir / plant_name / "healthy" / f"{photo['id']}.jpg"
                            with open(filename, "wb") as f:
                                f.write(img_data)
                            
                            count += 1
                            if count >= 100:  # 100 images per species
                                break
                        except Exception as e:
                            print(f"⚠️  Failed to download {photo['id']}: {e}")
                            continue
                    
                if count >= 100:
                    break
            
            print(f"✓ Downloaded {count} images of {plant_name}")
    
    def create_class_mappings(self):
        """Create JSON mappings for all classes"""
        
        disease_classes = {
            # Fungal
            0: {"name": "Powdery Mildew", "type": "fungal", "remedies": ["Neem oil spray", "Sulfur dust"]},
            1: {"name": "Leaf Spot", "type": "fungal", "remedies": ["Remove infected leaves", "Copper fungicide"]},
            2: {"name": "Rust", "type": "fungal", "remedies": ["Improve air circulation", "Fungicide spray"]},
            3: {"name": "Root Rot", "type": "fungal", "remedies": ["Reduce watering", "Repot in dry soil"]},
            4: {"name": "Damping Off", "type": "fungal", "remedies": ["Avoid overwatering", "Fungicide for seedlings"]},
            
            # Bacterial
            5: {"name": "Bacterial Leaf Spot", "type": "bacterial", "remedies": ["Remove affected leaves", "Copper spray"]},
            6: {"name": "Bacterial Wilt", "type": "bacterial", "remedies": ["Remove plant", "Disinfect soil"]},
            7: {"name": "Blight", "type": "bacterial", "remedies": ["Prune affected areas", "Bactericide spray"]},
            
            # Viral
            8: {"name": "Mosaic Virus", "type": "viral", "remedies": ["No cure - remove plant", "Prevent aphids"]},
            9: {"name": "Yellow Vein Virus", "type": "viral", "remedies": ["Remove plant", "Control vectors"]},
            
            # Pest
            10: {"name": "Spider Mites", "type": "pest", "remedies": ["Spray water", "Insecticidal soap"]},
            11: {"name": "Mealybugs", "type": "pest", "remedies": ["Isolate plant", "Neem oil spray"]},
            12: {"name": "Scale", "type": "pest", "remedies": ["Pruning", "Horticultural oil"]},
            13: {"name": "Thrips", "type": "pest", "remedies": ["Insecticide spray", "Yellow sticky traps"]},
            
            # Environmental
            14: {"name": "Sunburn", "type": "environmental", "remedies": ["Move to shade", "Increase humidity"]},
            15: {"name": "Freezing Damage", "type": "environmental", "remedies": ["Move to warm location", "Cut damaged parts"]},
            16: {"name": "Salt Damage", "type": "environmental", "remedies": ["Use filtered water", "Flush soil"]},
            17: {"name": "Nutrient Burn", "type": "environmental", "remedies": ["Flush pot", "Reduce fertilizer"]},
            18: {"name": "Chlorosis", "type": "environmental", "remedies": ["Check pH", "Iron supplement"]},
            19: {"name": "Wilting", "type": "environmental", "remedies": ["Water properly", "Check temperature"]},
            20: {"name": "Healthy", "type": "healthy", "remedies": ["Maintain current care"]}
        }
        
        nutrient_classes = {
            0: {"symbol": "N", "name": "Nitrogen", "symptoms": ["Pale yellow leaves", "Stunted growth"]},
            1: {"symbol": "P", "name": "Phosphorus", "symptoms": ["Purple leaves", "Poor root development"]},
            2: {"symbol": "K", "name": "Potassium", "symptoms": ["Brown leaf edges", "Weak stems"]},
            3: {"symbol": "Fe", "name": "Iron", "symptoms": ["Yellow veins, green leaves", "Interveinal chlorosis"]},
            4: {"symbol": "Mg", "name": "Magnesium", "symptoms": ["Yellow between veins", "Bottom leaves affected first"]},
            5: {"symbol": "Ca", "name": "Calcium", "symptoms": ["Distorted new growth", "Blossom end rot"]},
            6: {"symbol": "Mn", "name": "Manganese", "symptoms": ["Tan spots", "Necrosis on leaves"]},
            7: {"symbol": "B", "name": "Boron", "symptoms": ["Thick leaves", "Cracked stems"]},
            8: {"name": "Healthy", "symptoms": ["Normal growth"]}
        }
        
        # Save mappings
        with open(self.data_dir / "disease_classes.json", "w") as f:
            json.dump(disease_classes, f, indent=2)
        
        with open(self.data_dir / "nutrient_classes.json", "w") as f:
            json.dump(nutrient_classes, f, indent=2)
        
        print("✓ Created class mappings")
        
    def split_train_val_test(self, train_ratio=0.7, val_ratio=0.15):
        """Split dataset into train/val/test"""
        import random
        from sklearn.model_selection import train_test_split
        
        print("📊 Splitting dataset...")
        
        base_dir = self.plant_dir
        
        for split in ["train", "val", "test"]:
            (base_dir / split).mkdir(exist_ok=True)
        
        # Iterate all classes
        for class_dir in base_dir.iterdir():
            if class_dir.is_dir() and class_dir.name not in ["train", "val", "test"]:
                images = list(class_dir.rglob("*.jpg")) + list(class_dir.rglob("*.png"))
                
                # Shuffle
                random.shuffle(images)
                
                # Split
                train_size = int(len(images) * train_ratio)
                val_size = int(len(images) * val_ratio)
                
                for idx, img_path in enumerate(images):
                    if idx < train_size:
                        split = "train"
                    elif idx < train_size + val_size:
                        split = "val"
                    else:
                        split = "test"
                    
                    # Create subdirectory
                    split_class_dir = base_dir / split / class_dir.name
                    split_class_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Copy image
                    dst = split_class_dir / img_path.name
                    shutil.copy(img_path, dst)
        
        print("✓ Dataset split complete")


# RUN DATASET PREP
if __name__ == "__main__":
    builder = HousePlantDatasetBuilder()
    print("🌱 Building Jiva Plants dataset...")
    
    # Uncomment to download (requires kaggle API key)
    # builder.download_plantvillage()
    # builder.download_inaturalist_houseplants()
    
    builder.create_class_mappings()
    # builder.split_train_val_test()
    
    print("✅ Dataset preparation complete!")