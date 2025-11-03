from ast import literal_eval
import os
from pathlib import Path
from pydoc import text
from re import split

from numpy.ma import count
import pandas as pd
import classification_model as model

from image_extraction_helper import draw_bounding_box

FOLDER_PATH = "/Users/bikashgiri/Downloads/annotations_for_students/miceid_20251003144839"

train_csv = "outputs/train.csv"
csv_file = []


def extract_label_from_textfile(file):
    
     with open(file, "r") as f:
        text = f.read().split()
        #convert text to float since we have number in text file
        label_in_number = list(map(float, text))
        for number in label_in_number:
            if number > 4 :
                return number
                print(number)
    
        return 0
                # print(label_in_number)

def extract_row_information(row):
    
    """
        Extract row information
    
    Parameters:
    - row: row of current loop 
    """

    label = Path(FOLDER_PATH)/"labels"
    patch_label = 0

    
    # Check if folder exists
    if not os.path.exists(label):
        print(f"Folder '{label}' does not exist!")
        return
    
    # Get all files in the folder
    files = os.listdir(label)

    imageName = fr"{row.image_id}".replace("\\","/")
    print(imageName)


    # get last path of a file wihout extension
    path_without_extension = Path(imageName).stem
    path_suffix = Path(path_without_extension).name
    print(path_suffix + "suffix_name")

    #Add .txt and .png extension to our filename as our row path contains .png with different foloder structure
    file_with_txt = path_without_extension + ".txt"
    file_with_image = path_without_extension + ".png"
    textfile = Path(label) / file_with_txt

    patch_label = extract_label_from_textfile(textfile)




    image_path = Path(label)/ file_with_image 
    print("===================")
    print(image_path )
                
                
    splittedTailInfo = row.tailmark_info.split()
            
    if  "Empty DataFrame" in splittedTailInfo:
                    print("not found")
    else:
                    try:
                        # TODO This extracts only one mouse tail infromation.Remove the slice method 
                        slicedTailCoordinate = splittedTailInfo[17:21]
                        print(slicedTailCoordinate)
                        
                        # Check if we have enough elements and they can be converted to float
                        if len(slicedTailCoordinate) >= 4:
                            coordinate = list(map(float,slicedTailCoordinate))
                            print(f"Coordinates: {coordinate}")
                            
                            draw_bounding_box(
                                image_path= image_path,
                                coordinates = coordinate,
                                color=(0, 255, 0),  # Green in BGR
                                thickness=3,
                                label="Tail-Patches",
                                class_label= patch_label
                            )
                            
                        else:
                            print(f"Not enough coordinate data: {slicedTailCoordinate}")
                    except (ValueError, IndexError) as e:
                        print(f"Error processing coordinates: {e}")


    
    
    # for filename in files:
    #     file = Path(label)/ filename 
    #     # Get file extension
    #     file_ext = os.path.splitext(filename)[1].lower()
        
    #     if file_ext == '.txt':
    #         # file = Path(label)/ filename  
    #         # patch_label = extract_label_from_textfile(file) 
    #         print("")
            
    #     elif file_ext == '.png':
    #         imageName = Path(row.image_id).name 

    #         if imageName.endswith(filename):
    #             image_path = Path(label)/ filename 
                
                
    #             splittedTailInfo = row.tailmark_info.split()
            
    #             if  "Empty DataFrame" in splittedTailInfo:
    #                 print("not found")
    #             else:
    #                 try:
    #                     # TODO This extracts only one mouse tail infromation.Remove the slice method 
    #                     slicedTailCoordinate = splittedTailInfo[17:21]
    #                     print(slicedTailCoordinate)
                        
    #                     # Check if we have enough elements and they can be converted to float
    #                     if len(slicedTailCoordinate) >= 4:
    #                         coordinate = list(map(float,slicedTailCoordinate))
    #                         print(f"Coordinates: {coordinate}")
    #                         draw_bounding_box(
    #                             image_path= image_path,
    #                             coordinates = coordinate,
    #                             color=(0, 255, 0),  # Green in BGR
    #                             thickness=3,
    #                             label="Tail-Patches",
    #                             class_label= patch_label
    #                         )
    #                     else:
    #                         print(f"Not enough coordinate data: {slicedTailCoordinate}")
    #                 except (ValueError, IndexError) as e:
    #                     print(f"Error processing coordinates: {e}")
                        

    #     else:
    #         print(f"  → Other file type: {file_ext}")

def create_new_CSV():
    import csv
    # Create folder if it doesn't exist
    os.makedirs('outputs', exist_ok=True)

    # File path inside outputs folder

    # Define the header
    header = ['BoundingBox', 'Patch', 'PatchLabel']
    

    # Create CSV file and write header
    with open(train_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)


    with open(train_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        

def iterate_over_CSV():

    csvPath = Path(FOLDER_PATH) / "train.csv"
    df = pd.read_csv(csvPath)
    df = df.reset_index(drop=True)

    for row in df.itertuples():
     extract_row_information(row)

    



def main(): 
    create_new_CSV()   
    iterate_over_CSV()


     # Add your code here

# Using the special variable 
# __name__
if __name__=="__main__":
    main()



