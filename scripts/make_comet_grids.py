from comet_ml.api import API
from pathlib import Path
from PIL import Image, ImageOps
from tqdm import tqdm
import re
import math
import io
from collections import defaultdict

# ---------- Config ----------
API_KEY     = "y2tkTNGtg7kP3HX9mfdy8JHaM"
WORKSPACE   = "ioannisgkinis"
PROJECT     = "hcmr-ai"         # e.g., "med-wav" or similar
EXPERIMENT_KEYS = ["d39937d34aeb4544800345356e3106b8"]                 # e.g., ["abc1234567890...", "def0987..."] or None for all in project
TAG_FILTER  = None                     # e.g., "val" or None
FILENAME_REGEX = r"map__season__.*?__(mae|rmse|bias)__.*?\.svg$"  # Look for map__season__*__(mae|rmse|bias)__*.svg files
MAX_PER_EXP = 40                       # cap number of images per experiment
OUT_DIR     = Path("comet_grids")
GRID_COLS   = 6                        # number of columns in grid
CELL_SIZE   = (600, 450)               # (width, height) each thumbnail

OUT_DIR.mkdir(parents=True, exist_ok=True)

api = API(api_key=API_KEY)

def extract_metric_from_filename(filename):
    """
    Extract metric from filename like 'map__season__winter__bias__VHM0.svg'
    Returns the metric part (e.g., 'VHM0') or None if pattern doesn't match
    """
    # Pattern to match files like map__season__winter__bias__VHM0.svg
    pattern = r'map__.*?__.*?__.*?__(.+)\.svg$'
    match = re.search(pattern, filename)
    if match:
        return match.group(1)
    return None

def list_experiments(workspace, project, experiment_keys=None):
    if experiment_keys:
        # Fetch specific experiments by key
        exps = []
        for k in experiment_keys:
            try:
                exps.append(api.get_experiment(workspace, project, k))
            except Exception as e:
                print(f"⚠️ Could not fetch experiment {k}: {e}")
        return exps
    else:
        return api.get_experiments(workspace=workspace, project_name=project)

def should_keep(asset, tag_filter=None, filename_regex=None):
    if asset.get("type") not in ("image", "image-snapshot"):
        return False
    
    fname = asset.get("fileName") or ""
    
    # Include SVG files since they are the actual plots we want
    # We'll handle SVG conversion in download_image function
    
    if tag_filter:
        tags = asset.get("tags") or []
        if tag_filter not in tags:
            return False
    if filename_regex:
        if re.search(filename_regex, fname) is None:
            return False
    return True

def download_image(asset_bytes):
    # Returns a PIL.Image from raw bytes, converting to RGB and removing alpha (helps with uniform grids)
    try:
        im = Image.open(io.BytesIO(asset_bytes))
        if im.mode in ("RGBA", "LA"):
            # flatten alpha on white background
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            return bg
        return im.convert("RGB")
    except Exception as e:
        print(f"⚠️ Failed to process image: {e}")
        return None

def handle_svg_file(asset_bytes, filename):
    """
    Handle SVG files by converting them to PNG using cairosvg
    """
    try:
        import cairosvg
        
        # Convert SVG to PNG using cairosvg
        png_data = cairosvg.svg2png(bytestring=asset_bytes, output_width=320, output_height=320)
        
        # Convert PNG data to PIL Image
        img = Image.open(io.BytesIO(png_data))
        img = img.convert("RGB")
        
        return img
        
    except Exception as e:
        print(f"⚠️ Failed to convert SVG {filename}: {e}")
        # Fallback to placeholder if conversion fails
        img = Image.new("RGB", (320, 320), (240, 240, 240))
        from PIL import ImageDraw, ImageFont
        
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        # Add error message
        lines = [
            f"SVG: {filename}",
            "Conversion failed:",
            str(e)[:30] + "..."
        ]
        
        y_offset = 50
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            x = (320 - text_width) // 2
            draw.text((x, y_offset), line, fill=(0, 0, 0), font=font)
            y_offset += 25
            
        return img

def fetch_images_for_experiment(exp, max_items=MAX_PER_EXP, tag_filter=None, filename_regex=None):
    assets = exp.get_asset_list()
    print(f"🔍 Found {len(assets)} total assets in experiment {exp.key[:8]}")
    
    # Debug: show first few assets
    for i, asset in enumerate(assets[:5]):
        print(f"  Asset {i+1}: {asset.get('fileName', 'NO_NAME')} (type: {asset.get('type', 'NO_TYPE')})")
    
    # Keep only image assets; sort by 'step' then 'createdAt'
    filtered = [a for a in assets if should_keep(a, tag_filter, filename_regex)]
    print(f"📸 After filtering: {len(filtered)} image assets")
    
    filtered.sort(key=lambda a: (a.get("step", 0), a.get("createdAt", 0)))
    if max_items:
        filtered = filtered[-max_items:]  # last N

    results = []
    for a in tqdm(filtered, desc=f"Downloading {exp.key[:8]}"):
        try:
            raw = exp.get_asset(a["assetId"])
            filename = a.get("fileName") or f"{a['assetId']}.png"
            
            # Handle SVG files differently
            if filename.lower().endswith('.svg'):
                img = handle_svg_file(raw, filename)
            else:
                img = download_image(raw)
                
            if img is not None:  # Only add if image was successfully processed
                results.append({
                    "image": img,
                    "filename": filename,
                    "step": a.get("step"),
                    "metadata": a,
                    "experiment_key": exp.key,
                    "experiment_name": exp.get_metadata().get("experimentName") or exp.key[:8]
                })
        except Exception as e:
            print(f"⚠️ Failed to download {a.get('fileName') or a['assetId']}: {e}")
    return results

def make_grid(images, cols=GRID_COLS, cell_size=CELL_SIZE, pad=6, pad_color=(240,240,240)):
    """
    images: list[PIL.Image]
    returns a PIL.Image grid
    """
    if not images:
        return None
    # Thumbnail each image preserving aspect ratio inside the cell
    thums = []
    for im in images:
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

def create_metric_summary(metric_images):
    """
    Create a summary report of all files grouped by metric
    """
    summary_path = OUT_DIR / "metric_summary_report.txt"
    
    with open(summary_path, 'w') as f:
        f.write("METRIC GRID SUMMARY REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        for metric, images in metric_images.items():
            f.write(f"METRIC: {metric}\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total files: {len(images)}\n")
            f.write(f"Experiments: {len(set(img['experiment_key'] for img in images))}\n\n")
            
            # Group by experiment
            by_experiment = {}
            for img in images:
                exp_name = img['experiment_name']
                if exp_name not in by_experiment:
                    by_experiment[exp_name] = []
                by_experiment[exp_name].append(img['filename'])
            
            for exp_name, filenames in by_experiment.items():
                f.write(f"Experiment: {exp_name}\n")
                for filename in sorted(filenames):
                    f.write(f"  - {filename}\n")
                f.write("\n")
            
            f.write("\n" + "=" * 50 + "\n\n")
    
    print(f"📋 Created summary report: {summary_path}")
    return summary_path

def download_svg_files_as_png(metric_images):
    """
    Download SVG files and convert them to PNG files organized by metric
    """
    for metric, images in metric_images.items():
        # Create folder for this metric
        metric_dir = OUT_DIR / f"metric_{metric}"
        metric_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Downloading and converting {len(images)} files for metric {metric} to {metric_dir}")
        
        for i, img_data in enumerate(images):
            try:
                # Get the raw SVG data
                exp = api.get_experiment(WORKSPACE, PROJECT, img_data["experiment_key"])
                raw = exp.get_asset(img_data["metadata"]["assetId"])
                
                # Create filename with experiment info (change extension to .png)
                exp_name = img_data["experiment_name"].replace(" ", "_").replace("/", "_")
                original_filename = img_data['filename']
                png_filename = original_filename.replace('.svg', '.png')
                filename = f"{i+1:02d}_{exp_name}_{png_filename}"
                file_path = metric_dir / filename
                
                # Convert SVG to PNG using cairosvg
                import cairosvg
                png_data = cairosvg.svg2png(bytestring=raw, output_width=800, output_height=600)
                
                # Save the PNG file
                with open(file_path, 'wb') as f:
                    f.write(png_data)
                
                print(f"  ✅ Downloaded and converted: {filename}")
                
            except Exception as e:
                print(f"  ❌ Failed to download/convert {img_data['filename']}: {e}")
        
        print(f"📂 Metric {metric} PNG files saved to: {metric_dir}\n")

def main():
    exps = list_experiments(WORKSPACE, PROJECT, EXPERIMENT_KEYS)
    if not exps:
        print("No experiments found.")
        return

    # Dictionary to group images by metric
    metric_images = defaultdict(list)
    
    for exp in exps:
        imgs = fetch_images_for_experiment(
            exp, max_items=MAX_PER_EXP,
            tag_filter=TAG_FILTER,
            filename_regex=FILENAME_REGEX
        )
        if not imgs:
            print(f"ℹ️ No matching images for experiment {exp.key}")
            continue

        # Group images by metric
        for img_data in imgs:
            filename = img_data["filename"]
            metric = extract_metric_from_filename(filename)
            
            if metric:
                metric_images[metric].append(img_data)
            else:
                # If no metric pattern found, use filename as metric
                metric_images[filename].append(img_data)

    # Skip grid creation for now, focus on downloading files
    print("📥 Downloading SVG files instead of creating grids...")

    # Summary
    print(f"\n🎯 Summary:")
    print(f"Total metrics found: {len(metric_images)}")
    for metric, images in metric_images.items():
        print(f"  - {metric}: {len(images)} images")

    # Create a summary report of all files grouped by metric
    create_metric_summary(metric_images)

    # Download SVG files and convert to PNG
    download_svg_files_as_png(metric_images)

if __name__ == "__main__":
    main()
