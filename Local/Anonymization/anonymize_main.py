# TODO: Main script to anonymize audio and store in a new folder.
import os,sys
sys.path.append("./Local/")
sys.path.append("/mnt/d/projects/speaker-anonymization-gan/")

import pandas as pd
from tqdm import tqdm
import mcadams
import timeit
from Utils import util
import app
from app import ss_anonymize

def ano_per_dataset(metadata_path:str, anonymize:str, ano_metadata_path:str=None):

    assert os.path.exists(metadata_path), "Required metadata file is not in this location"
    assert anonymize in ['og','mcadams','ss'], "Input anonymize approach is not supported"

    if ano_metadata_path is None:
        ano_metadata_path = os.path.join(os.path.dirname(metadata_path),'metadata_v2_%s.csv'%(anonymize))

    df_md = pd.read_csv(metadata_path) # load metadata
    assert 'audio_path' in df_md.columns.values, "Required info missing in metadata"

    all_audio_path = df_md['audio_path'].tolist()
    ano_audio_path = []
    error_files=[]

    print('Start anonymization | %s ...'%(anonymize))

    starttime = timeit.default_timer()

    for ad_path in tqdm(all_audio_path):
        head, tail = os.path.split(ad_path)
        sample_id = util.remove_suffix(tail,'.wav')
        sample_id = util.remove_suffix(sample_id,'.flac')
        new_name = sample_id+'_%s.wav'%(anonymize)
        new_path = os.path.join(head,new_name) # anonymized audio path
        ano_audio_path.append(new_path)
        # anonymize audio
        try:
            if anonymize == 'mcadams':
                mcadams.anonym(input_path=ad_path,output_path=new_path)
            # TODO: add ss and mcadams-ss
            elif anonymize == 'ss':
                ss_anonymize(input_path=ad_path,output_path=new_path,model=app.VPInterface())
            
        except:
            print('Error occured in '+ad_path)
            error_files.append(ad_path)
            pass

    
    computetime = timeit.default_timer() - starttime
    ave_computetime = computetime/len(all_audio_path)

    print('Anonymization completed | %s'%(anonymize))
    print('Average time taken for anonymize one audio file is %f'%(ave_computetime))
    print(error_files)

    # Save feature path
    df_md_ano = df_md
    df_md_ano['audio_path'] = ano_audio_path
    df_md_ano.to_csv(ano_metadata_path)

    return df_md_ano


# %%
if __name__ == '__main__':

    ano_per_dataset(

        metadata_path='/mnt/d/projects/COVID-datasets/CSS/label/metadata_v2.csv',
        anonymize='ss'

    )

    # ano_per_dataset(

    #     metadata_path='/mnt/d/projects/COVID-datasets/DiCOVA2/label/metadata_v2.csv',
    #     anonymize='ss'

    # )

    # ano_per_dataset(

    #     metadata_path='/mnt/d/projects/COVID-datasets/Cambridge_Task2/label/metadata_v2.csv',
    #     anonymize='ss'

    # )
