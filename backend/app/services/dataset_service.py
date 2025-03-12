import os
import json
import logging
import uuid
from datetime import datetime
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename

logger = logging.getLogger(__name__)

class DatasetService:
    def __init__(self):
        self.datasets_dir = os.path.join(os.getcwd(), 'app', 'data', 'datasets')
        self.metadata_file = os.path.join(self.datasets_dir, 'metadata.json')
        
        # Create directory if it doesn't exist
        os.makedirs(self.datasets_dir, exist_ok=True)
        
        # Create metadata file if it doesn't exist
        if not os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'w') as f:
                json.dump([], f)
    
    def get_all_datasets(self):
        """Get all datasets metadata"""
        try:
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading datasets metadata: {str(e)}")
            return []
    
    def get_dataset_by_id(self, dataset_id):
        """Get a dataset by ID"""
        try:
            # Get metadata
            metadata = self.get_all_datasets()
            dataset_meta = next((d for d in metadata if d['id'] == dataset_id), None)
            
            if not dataset_meta:
                return None
            
            # Load dataset file
            dataset_path = os.path.join(self.datasets_dir, f"{dataset_id}.json")
            with open(dataset_path, 'r') as f:
                dataset = json.load(f)
            
            # Combine metadata with dataset
            return {**dataset_meta, **dataset}
        except Exception as e:
            logger.error(f"Error getting dataset {dataset_id}: {str(e)}")
            return None
    
    def upload_dataset(self, file, name, description, user_id):
        """Upload a new dataset"""
        try:
            # Generate a unique ID
            dataset_id = str(uuid.uuid4())
            
            # Secure the filename
            filename = secure_filename(file.filename)
            
            # Process the file based on its extension
            if filename.endswith('.csv'):
                dataset = self._process_csv(file)
            elif filename.endswith('.json'):
                dataset = self._process_json(file)
            else:
                raise ValueError("Unsupported file format. Please upload CSV or JSON.")
            
            # Save dataset to file
            dataset_path = os.path.join(self.datasets_dir, f"{dataset_id}.json")
            with open(dataset_path, 'w') as f:
                json.dump(dataset, f)
            
            # Create metadata
            metadata = {
                'id': dataset_id,
                'name': name,
                'description': description,
                'filename': filename,
                'created_by': user_id,
                'created_at': datetime.now().isoformat(),
                'size': len(dataset['features']),
                'feature_count': len(dataset['features'][0]) if dataset['features'] else 0
            }
            
            # Update metadata file
            all_metadata = self.get_all_datasets()
            all_metadata.append(metadata)
            with open(self.metadata_file, 'w') as f:
                json.dump(all_metadata, f)
            
            return metadata
        except Exception as e:
            logger.error(f"Error uploading dataset: {str(e)}")
            raise
    
    def _process_csv(self, file):
        """Process a CSV file into features and labels"""
        try:
            # Read CSV
            df = pd.read_csv(file)
            
            # Assume the last column is the label
            features = df.iloc[:, :-1].values.tolist()
            labels = df.iloc[:, -1].values.tolist()
            
            return {
                'features': features,
                'labels': labels,
                'feature_names': df.columns[:-1].tolist(),
                'label_name': df.columns[-1]
            }
        except Exception as e:
            logger.error(f"Error processing CSV: {str(e)}")
            raise
    
    def _process_json(self, file):
        """Process a JSON file into features and labels"""
        try:
            # Read JSON
            data = json.load(file)
            
            # Check if the JSON has the expected format
            if 'features' not in data or 'labels' not in data:
                raise ValueError("JSON file must contain 'features' and 'labels' keys")
            
            return data
        except Exception as e:
            logger.error(f"Error processing JSON: {str(e)}")
            raise
    
    def delete_dataset(self, dataset_id):
        """Delete a dataset"""
        try:
            # Get metadata
            metadata = self.get_all_datasets()
            dataset_meta = next((d for d in metadata if d['id'] == dataset_id), None)
            
            if not dataset_meta:
                return False
            
            # Delete dataset file
            dataset_path = os.path.join(self.datasets_dir, f"{dataset_id}.json")
            if os.path.exists(dataset_path):
                os.remove(dataset_path)
            
            # Update metadata file
            updated_metadata = [d for d in metadata if d['id'] != dataset_id]
            with open(self.metadata_file, 'w') as f:
                json.dump(updated_metadata, f)
            
            return True
        except Exception as e:
            logger.error(f"Error deleting dataset {dataset_id}: {str(e)}")
            return False
    
    def get_dataset_statistics(self, dataset_id):
        """Get statistics for a dataset"""
        try:
            dataset = self.get_dataset_by_id(dataset_id)
            
            if not dataset:
                return None
            
            features = np.array(dataset['features'])
            labels = np.array(dataset['labels'])
            
            # Calculate statistics
            stats = {
                'feature_count': features.shape[1],
                'sample_count': features.shape[0],
                'feature_stats': []
            }
            
            # Calculate statistics for each feature
            for i in range(features.shape[1]):
                feature = features[:, i]
                feature_stats = {
                    'name': dataset.get('feature_names', [])[i] if i < len(dataset.get('feature_names', [])) else f"Feature {i}",
                    'min': float(np.min(feature)),
                    'max': float(np.max(feature)),
                    'mean': float(np.mean(feature)),
                    'std': float(np.std(feature))
                }
                stats['feature_stats'].append(feature_stats)
            
            # Calculate label statistics
            if len(labels.shape) == 1:
                # Single label
                stats['label_stats'] = {
                    'name': dataset.get('label_name', 'Label'),
                    'min': float(np.min(labels)),
                    'max': float(np.max(labels)),
                    'mean': float(np.mean(labels)),
                    'std': float(np.std(labels))
                }
            else:
                # Multiple labels
                stats['label_stats'] = []
                for i in range(labels.shape[1]):
                    label = labels[:, i]
                    label_stats = {
                        'name': f"Label {i}",
                        'min': float(np.min(label)),
                        'max': float(np.max(label)),
                        'mean': float(np.mean(label)),
                        'std': float(np.std(label))
                    }
                    stats['label_stats'].append(label_stats)
            
            return stats
        except Exception as e:
            logger.error(f"Error getting dataset statistics for {dataset_id}: {str(e)}")
            return None 