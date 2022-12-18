"""
Three datasets are included in this study:
1. ComParE 2022 CSS; 2. DiCOVA2; 3. Task-2 English set from Cambridge database.

As they have different ways to store the sound files and labels, 
we here write a script to create one .csv file for each dataset which includes three columns:
Audio_path|Label|Split

"""

def clean_CSS():

    # load train&devel&test labels (.csv) 
    
    # substitute sample_id with absolute file path

    # add one column to indicate split (train/valid/test)

    # concatenate three tables into one

    return 0


def clean_DiCOVA2():

    # load metadata (.csv) and pre-defined split (.csv)

    # substitute sample_id with absolute file path

    # add one column to indicate split (train/valid/test)

    return 0


def clean_Cambridge():

    # load labels (.csv)

    # change column names

    return 0