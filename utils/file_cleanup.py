import os
import time
import shutil
from datetime import datetime, timedelta
import threading

class FileCleanup:
    """
    Automatic cleanup of old processed files
    """
    
    def __init__(self, processed_dir="processed", uploads_dir="uploads", max_age_hours=24):
        self.processed_dir = processed_dir
        self.uploads_dir = uploads_dir
        self.max_age_hours = max_age_hours
        self.cleanup_interval = 3600  # Run every hour
        self.running = False
        self.cleanup_thread = None
    
    def start_cleanup_scheduler(self):
        """Start automatic cleanup scheduler"""
        if self.running:
            print("⚠️ Cleanup scheduler already running")
            return
        
        self.running = True
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
        print(f"✅ File cleanup scheduler started (runs every {self.cleanup_interval//3600} hour)")
        print(f"   Files older than {self.max_age_hours} hours will be deleted")
    
    def stop_cleanup_scheduler(self):
        """Stop automatic cleanup scheduler"""
        self.running = False
        print("🛑 File cleanup scheduler stopped")
    
    def _cleanup_loop(self):
        """Background loop that runs cleanup periodically"""
        while self.running:
            try:
                self.cleanup_old_files()
            except Exception as e:
                print(f"❌ Cleanup error: {e}")
            
            # Wait for next cleanup cycle
            time.sleep(self.cleanup_interval)
    
    def cleanup_old_files(self):
        """Remove files older than max_age_hours"""
        current_time = time.time()
        max_age_seconds = self.max_age_hours * 3600
        
        deleted_count = 0
        freed_space_mb = 0
        
        print(f"\n🗑️ Running file cleanup... (removing files older than {self.max_age_hours}h)")
        
        # Cleanup processed directory
        if os.path.exists(self.processed_dir):
            deleted, freed = self._cleanup_directory(self.processed_dir, current_time, max_age_seconds)
            deleted_count += deleted
            freed_space_mb += freed
        
        # Cleanup uploads directory
        if os.path.exists(self.uploads_dir):
            deleted, freed = self._cleanup_directory(self.uploads_dir, current_time, max_age_seconds)
            deleted_count += deleted
            freed_space_mb += freed
        
        if deleted_count > 0:
            print(f"✅ Cleanup complete: Deleted {deleted_count} items, freed {freed_space_mb:.2f} MB")
        else:
            print(f"✅ Cleanup complete: No old files to delete")
    
    def _cleanup_directory(self, directory, current_time, max_age_seconds):
        """Cleanup a specific directory"""
        deleted_count = 0
        freed_space_mb = 0
        
        try:
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                
                try:
                    # Get file/folder age
                    item_age = current_time - os.path.getmtime(item_path)
                    
                    # Check if old enough to delete
                    if item_age > max_age_seconds:
                        # Get size before deletion
                        if os.path.isfile(item_path):
                            size_bytes = os.path.getsize(item_path)
                        else:
                            size_bytes = self._get_folder_size(item_path)
                        
                        freed_space_mb += size_bytes / (1024 * 1024)
                        
                        # Delete
                        if os.path.isfile(item_path):
                            os.remove(item_path)
                            print(f"  🗑️ Deleted file: {item}")
                        else:
                            shutil.rmtree(item_path)
                            print(f"  🗑️ Deleted folder: {item}")
                        
                        deleted_count += 1
                
                except Exception as e:
                    print(f"  ⚠️ Could not delete {item}: {e}")
                    continue
        
        except Exception as e:
            print(f"  ⚠️ Error accessing directory {directory}: {e}")
        
        return deleted_count, freed_space_mb
    
    def _get_folder_size(self, folder_path):
        """Calculate total size of a folder"""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(folder_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    try:
                        total_size += os.path.getsize(filepath)
                    except:
                        pass
        except:
            pass
        return total_size
    
    def cleanup_specific_job(self, job_id):
        """Manually cleanup a specific job"""
        job_dir = os.path.join(self.processed_dir, job_id)
        
        if os.path.exists(job_dir):
            try:
                size_mb = self._get_folder_size(job_dir) / (1024 * 1024)
                shutil.rmtree(job_dir)
                print(f"✅ Deleted job {job_id} ({size_mb:.2f} MB freed)")
                return True
            except Exception as e:
                print(f"❌ Could not delete job {job_id}: {e}")
                return False
        else:
            print(f"⚠️ Job {job_id} not found")
            return False
    
    def get_storage_stats(self):
        """Get current storage statistics"""
        stats = {
            "processed_files": 0,
            "processed_size_mb": 0,
            "uploads_files": 0,
            "uploads_size_mb": 0,
            "total_size_mb": 0
        }
        
        try:
            if os.path.exists(self.processed_dir):
                stats["processed_files"] = len(os.listdir(self.processed_dir))
                stats["processed_size_mb"] = self._get_folder_size(self.processed_dir) / (1024 * 1024)
            
            if os.path.exists(self.uploads_dir):
                stats["uploads_files"] = len(os.listdir(self.uploads_dir))
                stats["uploads_size_mb"] = self._get_folder_size(self.uploads_dir) / (1024 * 1024)
            
            stats["total_size_mb"] = stats["processed_size_mb"] + stats["uploads_size_mb"]
        
        except Exception as e:
            print(f"Error getting storage stats: {e}")
        
        return stats