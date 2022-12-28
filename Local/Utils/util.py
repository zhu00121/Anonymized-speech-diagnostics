import json
import os
import pickle as pkl

def save_as_json(data,save_path):
    """
    Save results as .json file.
    """
    assert not os.path.exists(save_path), "File existing with the same name."
    with open(save_path, "w") as outfile:
        json.dump(data, outfile)
    
    return 0


def load_json(save_path:str):
    """
    Load a saved .json file.
    """
    assert os.path.exists(save_path), "json file does not exist!"
    with open(save_path, 'r') as f:
        saved_json = json.load(f)
    return saved_json


def save_as_pkl(save_path:str,variable):

    """
    Save result as a pickle file.
    """
    assert not os.path.exists(save_path), 'Existing file with same name.'
    
    with open(save_path, "wb") as outfile:
        pkl.dump(variable, outfile, pkl.HIGHEST_PROTOCOL)
        
    return 0


def load_pkl(save_path:str):

    """
    Load saved pickle file.
    """
    assert os.path.exists(save_path), "pkl file does not exist!"
    with open(save_path, "rb") as infile:
        saved_fea = pkl.load(infile)

    return saved_fea


def remove_suffix(input_string, suffix):
    if suffix and input_string.endswith(suffix):
        return input_string[:-len(suffix)]
    return input_string


def convertType(val):
	def subutil(val):
		try: 
			val = int(val)
		except:
			try:
				val = float(val)
			except:
				if val in ['True', 'TRUE', 'true']:
					val = True
				elif val in ['False','FALSE','false']:
					val = False
				elif val in ['None']:
					val = None
				else:
					val = val
		return val

	if ',' in val:
		val = val.split(',')
		val = [subutil(item) for item in val]
	else:
		val = subutil(val)
	return val