from ast import literal_eval
import os
from pathlib import Path
from re import split

from numpy.ma import count
import pandas as pd

from image_extraction_helper import draw_bounding_box

folder_path = "/Users/bikashgiri/Downloads/annotations_for_students/miceid_20251003144839"



def extract_row_information(row):
    
    """
    Loop through all files in a folder.
    
    Parameters:
    - folder_path: Path to the folder containing files
    """

    labelledFilePath = Path(folder_path)/"labels"

    
    # Check if folder exists
    if not os.path.exists(labelledFilePath):
        print(f"Folder '{labelledFilePath}' does not exist!")
        return
    
    # Get all files in the folder
    files = os.listdir(labelledFilePath)
    
    for filename in files:
         
        # Get file extension
        file_ext = os.path.splitext(filename)[1].lower()
        
        if file_ext == '.txt':
           # TODO extract tail mark label from .txt file 
            a= "asdf"
            
        elif file_ext == '.png':
            imageName = Path(row.image_id).name 

            if imageName.endswith(filename):
                image_path = Path(labelledFilePath)/ filename 
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
                                label="Tail-Patches"
                            )
                        else:
                            print(f"Not enough coordinate data: {slicedTailCoordinate}")
                    except (ValueError, IndexError) as e:
                        print(f"Error processing coordinates: {e}")
            
        else:
            print(f"  → Other file type: {file_ext}")

def iterate_over_CSV():

    csvPath = Path(folder_path) / "train.csv"
    df = pd.read_csv(csvPath)
    df = df.reset_index(drop=True)

    for row in df.itertuples():
        extract_row_information(row)
    



def main():    
    iterate_over_CSV()
     # Add your code here

# Using the special variable 
# __name__
if __name__=="__main__":
    main()



