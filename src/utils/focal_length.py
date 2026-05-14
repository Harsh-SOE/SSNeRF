import exifread

def extract_focal_length_data(image_path: str):
    try:
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f)

        if not tags:
            print("No EXIF data found. (This image may have been resized, compressed, or stripped of metadata).")
            return

        make_tag = 'Image Make'
        model_tag = 'Image Model'
        focal_physical_tag = 'EXIF FocalLength'
        focal_35mm_tag = 'EXIF FocalLengthIn35mmFilm'

        if make_tag in tags and model_tag in tags:
            print(f"Camera Model: {tags[make_tag]} {tags[model_tag]}")
        else:
            print("Camera Model: Unknown")

        if focal_physical_tag in tags:
            phys_val = tags[focal_physical_tag]
            try:
                phys_float = float(eval(str(phys_val)))
                print(f"Physical Focal Length: {phys_float:.2f} mm")
            except:
                print(f"Physical Focal Length: {phys_val} mm")
        else:
            print("Physical Focal Length: NOT FOUND")

        if focal_35mm_tag in tags:
            print(f"35mm Equivalent: {tags[focal_35mm_tag]} mm")
        else:
            print("35mm Equivalent: NOT FOUND")

    except FileNotFoundError:
        print(f"Error: Could not find the file '{image_path}'. Check your path.")
    except Exception as e:
        print(f"An error occurred: {e}")
