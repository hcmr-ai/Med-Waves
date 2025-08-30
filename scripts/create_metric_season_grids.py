from pathlib import Path
from PIL import Image, ImageOps, ImageDraw, ImageFont
import re
import math
import json
from collections import defaultdict

def add_label_to_image(img, label, position="bottom"):
    """
    Add a text label to an image
    """
    try:
        # Try to use a default font, fall back to basic if not available
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        except:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    # Create a copy of the image to avoid modifying the original
    img_with_label = img.copy()
    draw = ImageDraw.Draw(img_with_label)
    
    # Get text size
    bbox = draw.textbbox((0, 0), label, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Calculate position
    if position == "bottom":
        x = (img.width - text_width) // 2
        y = img.height - text_height - 10
    elif position == "top":
        x = (img.width - text_width) // 2
        y = 10
    else:  # center
        x = (img.width - text_width) // 2
        y = (img.height - text_height) // 2
    
    # Add background rectangle for better readability
    padding = 5
    draw.rectangle([x - padding, y - padding, x + text_width + padding, y + text_height + padding], 
                   fill=(255, 255, 255, 200))
    
    # Draw text
    draw.text((x, y), label, fill=(0, 0, 0), font=font)
    
    return img_with_label

def make_grid(images, labels, cols=2, cell_size=(600, 450), pad=15, pad_color=(240,240,240)):
    """
    Create a grid from a list of PIL Images with labels
    """
    if not images:
        return None
    
    # Add labels to images
    labeled_images = []
    for img, label in zip(images, labels):
        labeled_img = add_label_to_image(img, label, position="bottom")
        labeled_images.append(labeled_img)
    
    # Thumbnail each image preserving aspect ratio inside the cell
    thums = []
    for im in labeled_images:
        im = ImageOps.contain(im, cell_size)  # fit inside cell
        # add padding canvas to exact cell_size
        canvas = Image.new("RGB", cell_size, (255,255,255))
        x = (cell_size[0] - im.width) // 2
        y = (cell_size[1] - im.height) // 2
        canvas.paste(im, (x, y))
        thums.append(canvas)

    rows = math.ceil(len(thums) / cols)
    grid_w = cols * cell_size[0] + (cols+1) * pad
    grid_h = rows * cell_size[1] + (rows+1) * pad
    grid = Image.new("RGB", (grid_w, grid_h), pad_color)

    i = 0
    for r in range(rows):
        for c in range(cols):
            if i >= len(thums):
                break
            x = pad + c * (cell_size[0] + pad)
            y = pad + r * (cell_size[1] + pad)
            grid.paste(thums[i], (x, y))
            i += 1
    return grid

def analyze_file_distribution(comet_grids_dir):
    """
    Analyze the distribution of files across metrics, seasons, and experiments
    """
    analysis = {
        "total_files": 0,
        "metrics": {},
        "seasons": {},
        "experiments": {},
        "season_metric_combinations": {},
        "experiment_metric_combinations": {},
        "missing_combinations": []
    }
    
    # Find metric directories
    metric_dirs = [d for d in comet_grids_dir.iterdir() if d.is_dir() and d.name.startswith("metric_")]
    
    for metric_dir in metric_dirs:
        metric_name = metric_dir.name.replace("metric_", "")
        png_files = list(metric_dir.glob("*.png"))
        
        analysis["total_files"] += len(png_files)
        
        if metric_name not in analysis["metrics"]:
            analysis["metrics"][metric_name] = 0
        analysis["metrics"][metric_name] += len(png_files)
        
        for png_file in png_files:
            # Extract information from filename
            match = re.search(r'map__season__(\w+)__(mae|rmse|bias)__', png_file.name)
            exp_match = re.search(r'(\d+)_(.+?)_map__', png_file.name)
            
            if match and exp_match:
                season = match.group(1)
                metric_type = match.group(2)
                exp_name = exp_match.group(2)
                
                # Clean up experiment name
                if "DeltaCorrector" in exp_name:
                    exp_label = "DeltaCorrector"
                elif "random_regressor" in exp_name:
                    exp_label = "RandomRegressor"
                else:
                    exp_label = exp_name
                
                # Count seasons
                if season not in analysis["seasons"]:
                    analysis["seasons"][season] = 0
                analysis["seasons"][season] += 1
                
                # Count experiments
                if exp_label not in analysis["experiments"]:
                    analysis["experiments"][exp_label] = 0
                analysis["experiments"][exp_label] += 1
                
                # Count season-metric combinations
                key = f"{season}_{metric_type}"
                if key not in analysis["season_metric_combinations"]:
                    analysis["season_metric_combinations"][key] = 0
                analysis["season_metric_combinations"][key] += 1
                
                # Count experiment-metric combinations
                key = f"{exp_label}_{metric_type}"
                if key not in analysis["experiment_metric_combinations"]:
                    analysis["experiment_metric_combinations"][key] = 0
                analysis["experiment_metric_combinations"][key] += 1
    
    # Find missing combinations
    expected_combinations = []
    for season in ["winter", "spring", "summer", "autumn"]:
        for metric_type in ["bias", "rmse", "mae"]:
            for experiment in ["DeltaCorrector", "RandomRegressor"]:
                expected_combinations.append(f"{season}_{experiment}_{metric_type}")
    
    for combination in expected_combinations:
        season, experiment, metric_type = combination.split("_")
        key = f"{season}_{metric_type}"
        if key not in analysis["season_metric_combinations"] or analysis["season_metric_combinations"][key] == 0:
            analysis["missing_combinations"].append(combination)
    
    return analysis

def create_analysis_report(analysis, grids_dir):
    """
    Create a comprehensive analysis report
    """
    report_path = grids_dir / "analysis_report.md"
    
    with open(report_path, 'w') as f:
        f.write("# Wave Analysis Grid Report\n")
        f.write("## File Distribution Analysis\n\n")
        
        f.write(f"**Total Files Processed:** {analysis['total_files']}\n\n")
        
        f.write("### Metrics Distribution\n")
        for metric, count in analysis["metrics"].items():
            f.write(f"- **{metric}**: {count} files\n")
        f.write("\n")
        
        f.write("### Seasons Distribution\n")
        for season, count in analysis["seasons"].items():
            f.write(f"- **{season.capitalize()}**: {count} files\n")
        f.write("\n")
        
        f.write("### Experiments Distribution\n")
        for experiment, count in analysis["experiments"].items():
            f.write(f"- **{experiment}**: {count} files\n")
        f.write("\n")
        
        f.write("### Season-Metric Combinations\n")
        for combination, count in analysis["season_metric_combinations"].items():
            season, metric_type = combination.split("_")
            f.write(f"- **{season.capitalize()} {metric_type.upper()}**: {count} files\n")
        f.write("\n")
        
        f.write("### Experiment-Metric Combinations\n")
        for combination, count in analysis["experiment_metric_combinations"].items():
            experiment, metric_type = combination.split("_")
            f.write(f"- **{experiment} {metric_type.upper()}**: {count} files\n")
        f.write("\n")
        
        if analysis["missing_combinations"]:
            f.write("### Missing Combinations\n")
            f.write("The following combinations are missing:\n")
            for combination in analysis["missing_combinations"]:
                f.write(f"- {combination}\n")
            f.write("\n")
        
        f.write("## Grid Layout Summary\n\n")
        f.write("### Individual Metric Grids\n")
        f.write("- **Layout**: 2x2 (2 experiments per grid)\n")
        f.write("- **Content**: Single metric type per grid\n")
        f.write("- **Labels**: Experiment name + metric type\n\n")
        
        f.write("### Comprehensive Grids\n")
        f.write("- **Layout**: 2x3 (2 experiments x 3 metrics)\n")
        f.write("- **Content**: All metrics (Bias, RMSE, MAE) for both experiments\n")
        f.write("- **Labels**: Experiment name + metric type\n\n")
        
        f.write("## Recommendations for Presentation\n\n")
        f.write("1. **Start with comprehensive grids** to show overall patterns\n")
        f.write("2. **Use individual grids** for detailed metric comparisons\n")
        f.write("3. **Focus on seasonal patterns** - compare winter vs summer performance\n")
        f.write("4. **Highlight experiment differences** - DeltaCorrector vs RandomRegressor\n")
        f.write("5. **Consider metric relationships** - how Bias, RMSE, and MAE relate\n\n")
        
        f.write("## File Organization\n\n")
        f.write("All grids are saved in the `grids/` directory with descriptive filenames:\n")
        f.write("- `grid_[METRIC]_[TYPE]_[SEASON].png` - Individual metric grids\n")
        f.write("- `comprehensive_[SEASON]_all_experiments_all_metrics.png` - Comprehensive grids\n")
        f.write("- `*_metadata.txt` - Detailed information about each grid\n")
    
    print(f"📊 Analysis report created: {report_path}")
    return report_path

def create_metric_season_grids():
    """
    Create grids per metric and season with enhanced analysis
    """
    comet_grids_dir = Path("comet_grids")
    
    if not comet_grids_dir.exists():
        print("❌ comet_grids directory not found.")
        return
    
    # Find metric directories
    metric_dirs = [d for d in comet_grids_dir.iterdir() if d.is_dir() and d.name.startswith("metric_")]
    
    if not metric_dirs:
        print("❌ No metric directories found.")
        return
    
    # Create output directory for grids
    grids_dir = comet_grids_dir / "grids"
    grids_dir.mkdir(exist_ok=True)
    
    # Process each metric
    for metric_dir in sorted(metric_dirs):
        metric_name = metric_dir.name.replace("metric_", "")
        png_files = list(metric_dir.glob("*.png"))
        
        if not png_files:
            continue
            
        print(f"📊 Processing {metric_name}: {len(png_files)} files")
        
        # Group by metric type (mae, rmse, bias) and season
        metric_groups = {}
        for png_file in png_files:
            # Extract metric type and season from filename
            match = re.search(r'map__season__(\w+)__(mae|rmse|bias)__', png_file.name)
            if match:
                season = match.group(1)
                metric_type = match.group(2)
                key = f"{metric_type}_{season}"
                
                if key not in metric_groups:
                    metric_groups[key] = []
                metric_groups[key].append(png_file)
        
        # Create grids for each metric type and season
        for key, files in metric_groups.items():
            metric_type, season = key.split('_', 1)
            
            print(f"  🗺️ Creating grid for {metric_type.upper()} - {season.capitalize()}")
            
            # Load images and create labels
            images = []
            labels = []
            for png_file in sorted(files):
                try:
                    img = Image.open(png_file)
                    images.append(img)
                    
                    # Create label from filename
                    exp_match = re.search(r'(\d+)_(.+?)_map__', png_file.name)
                    if exp_match:
                        exp_name = exp_match.group(2)
                        # Clean up experiment name
                        if "DeltaCorrector" in exp_name:
                            exp_label = "DeltaCorrector"
                        elif "EDCDFCorrector" in exp_name:
                            exp_label = "EDCDFCorrector"
                        elif "DiffCorrector" in exp_name:
                            exp_label = "DiffCorrector"
                        elif "random_regressor" in exp_name:
                            exp_label = "RandomRegressor"
                        else:
                            exp_label = exp_name
                        
                        # Create more descriptive label
                        labels.append(f"{exp_label}\n{metric_type.upper()}")
                    else:
                        labels.append(f"Unknown\n{metric_type.upper()}")
                        
                except Exception as e:
                    print(f"    ❌ Failed to load {png_file.name}: {e}")
    
    # Create comprehensive grids with all metrics per season
    print("\n🎯 Creating comprehensive grids with all metrics per season...")
    create_comprehensive_grids(comet_grids_dir, grids_dir)
    
    print(f"\n🎯 All grids saved to: {grids_dir}")
    print("📋 You can now use these grid images in your Google Slides presentation!")
    print("📊 Check the analysis_report.md for detailed insights!")

def create_comprehensive_grids(comet_grids_dir, grids_dir):
    """
    Create comprehensive grids with all experiments and all metrics (bias, RMSE, MAE) per season
    """
    # Find all metric directories
    metric_dirs = [d for d in comet_grids_dir.iterdir() if d.is_dir() and d.name.startswith("metric_")]
    
    # Group files by season, experiment, and metric type
    season_groups = {}
    
    for metric_dir in metric_dirs:
        metric_name = metric_dir.name.replace("metric_", "")
        png_files = list(metric_dir.glob("*.png"))
        
        for png_file in png_files:
            # Extract season, metric type, and experiment from filename
            match = re.search(r'map__season__(\w+)__(mae|rmse|bias)__', png_file.name)
            exp_match = re.search(r'(\d+)_(.+?)_map__', png_file.name)
            
            if match and exp_match:
                season = match.group(1)
                metric_type = match.group(2)
                exp_name = exp_match.group(2)
                
                # Clean up experiment name
                if "DeltaCorrector" in exp_name:
                    exp_label = "DeltaCorrector"
                elif "random_regressor" in exp_name:
                    exp_label = "RandomRegressor"
                elif "DiffCorrector" in exp_name:
                    exp_label = "DiffCorrector"
                elif "EDCDFCorrector" in exp_name:
                    exp_label = "EDCDFCorrector"
                else:
                    exp_label = exp_name
                
                if season not in season_groups:
                    season_groups[season] = {}
                
                if exp_label not in season_groups[season]:
                    season_groups[season][exp_label] = {}
                
                if metric_type not in season_groups[season][exp_label]:
                    season_groups[season][exp_label][metric_type] = []
                
                season_groups[season][exp_label][metric_type].append(png_file)
    
    # Create comprehensive grids for each season
    for season, experiments in season_groups.items():
        print(f"  🗺️ Creating comprehensive grid for {season.capitalize()}")
        
        # Check if we have at least 2 experiments and all three metrics
        if len(experiments) >= 2:
            images = []
            labels = []
            # Create grid: experiments (rows) x 3 metrics (columns)
            # Get available experiments dynamically
            available_experiments = list(experiments.keys())
            for experiment in available_experiments:
                if experiment in experiments:
                    for metric_type in ['bias', 'rmse', 'mae']:
                        if metric_type in experiments[experiment] and experiments[experiment][metric_type]:
                            png_file = sorted(experiments[experiment][metric_type])[0]  # Take first file if multiple
                            try:
                                img = Image.open(png_file)
                                images.append(img)
                                labels.append(f"{experiment}\n{metric_type.upper()}")
                            except Exception as e:
                                print(f"    ❌ Failed to load {png_file.name}: {e}")
                        else:
                            # Create placeholder if missing
                            placeholder = Image.new("RGB", (600, 450), (200, 200, 200))
                            images.append(placeholder)
                            labels.append(f"{experiment}\n{metric_type.upper()}\n(Missing)")
                else:
                    # Create placeholders if experiment missing
                    for metric_type in ['bias', 'rmse', 'mae']:
                        placeholder = Image.new("RGB", (600, 450), (200, 200, 200))
                        images.append(placeholder)
                        labels.append(f"{experiment}\n{metric_type.upper()}\n(Missing)")
            
            if len(images) > 0:  # Check if we have any images
                # Determine grid layout based on number of experiments
                cols = 3  # Always 3 columns for the 3 metrics
                grid = make_grid(images, labels, cols=cols, cell_size=(500, 375))
            
            if grid:
                # Save grid
                grid_filename = f"comprehensive_{season}_all_experiments_all_metrics.png"
                grid_path = grids_dir / grid_filename
                grid.save(grid_path, quality=95)
                print(f"    ✅ Saved: {grid_filename}")
                    

if __name__ == "__main__":
    create_metric_season_grids()
