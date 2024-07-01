from seacrowd import SEACrowdConfigHelper
from seacrowd.utils.constants import Tasks
import pandas as pd
import datasets
from enum import Enum

NLU_TASK_LIST = {
    # Sentiment Analysis
    "wisesight_thai_sentiment_seacrowd_text",
    "wongnai_reviews_seacrowd_text",
    # "vlsp2016_sa_seacrowd_text", -- local dataset, will rectify later
    "typhoon_yolanda_tweets_seacrowd_text",
    # Topic Analysis
    "gklmip_newsclass_seacrowd_text",
    "indonesian_news_dataset_seacrowd_text",
    "uit_vion_seacrowd_text",
    # "total_defense_meme_topic_seacrowd_text",
    "sib_200_tha_Thai_seacrowd_text",
    # Reasoning
    # Standard Testing QA
    "m3exam_tha_seacrowd_qa",
    # "okapi_m_mmlu_ind_seacrowd_qa",
    # "okapi_m_mmlu_vie_seacrowd_qa",
    # # Cultural QA
    "xcopa_tha_seacrowd_qa",
    # Other QA
    "belebele_tha_thai_seacrowd_qa",
    # NLI
    "xnli.tha_seacrowd_pairs",
}

NLU_TASK_LIST_EXTERNAL = []

NLG_TASK_LIST = [
    # SUMMARIZATION
    "lr_sum_tha_seacrowd_t2t",
    # MACHINE TRANSLATION
    "ntrex_128_eng-US_tha_seacrowd_t2t",
    "flores200_tha_Thai_eng_Latn_seacrowd_t2t",
    # EXTRACTIVE ABSTRACTIVE QA
    # "idk_mrc_seacrowd_qa", -- many empty [] answers
    "mkqa_tha_seacrowd_qa",
]
SPEECH_TASK_LIST = list(dict.fromkeys([
    'commonvoice_120_tha_seacrowd_sptext',
    'fleurs_tha_seacrowd_sptext',
]))


FLORES200_TASK_LIST = []

VL_TASK_LIST = [
    "xm3600_th_seacrowd_imtext",
]


def load_vl_datasets():
    nc_conhelp = SEACrowdConfigHelper()
    cfg_name_to_dset_map = {}

    for config_name in VL_TASK_LIST:
        print(config_name)
        schema = config_name.split('_')[-1]
        language = config_name.split('_')[-3]
        con = nc_conhelp.for_config_name(config_name)
        cfg_name_to_dset_map[config_name] = (con.load_dataset(), language, list(con.tasks)[0])
    
    return cfg_name_to_dset_map


def load_nlu_datasets():
    nc_conhelp = SEACrowdConfigHelper()
    cfg_name_to_dset_map = {}

    for config_name in NLU_TASK_LIST:
        print(config_name)
        schema = config_name.split('_')[-1]
        con = nc_conhelp.for_config_name(config_name)
        cfg_name_to_dset_map[config_name] = (con.load_dataset(), list(con.tasks)[0])

    return cfg_name_to_dset_map


### Forget this for now
def load_external_nlu_datasets(lang='ind'):
    cfg_name_to_dset_map = {}  # {config_name: (datasets.Dataset, task_name)

    # hack, add new Task
    class NewTasks(Enum):
        COPA = "COPA"
        MABL = "MABL"
        MAPS = "MAPS"
        IndoStoryCloze = "IndoStoryCloze"
        IndoMMLU = "IndoMMLU"

    for task in NLU_TASK_LIST_EXTERNAL:
        if 'COPAL' in task:
            dset = datasets.load_dataset(task)
            cfg_name_to_dset_map[task] = (dset, NewTasks.COPA)
        elif 'MABL' in task:
            mabl_path = './mabl_data'
            subset = task.split('/')[-1]

            df = pd.read_csv(f'{mabl_path}/{subset}.csv')
            dset = datasets.Dataset.from_pandas(
                df.rename({'startphrase': 'premise', 'ending1': 'choice1', 'ending2': 'choice2', 'labels': 'label'},
                          axis='columns')
            )
            cfg_name_to_dset_map[task] = (datasets.DatasetDict({'test': dset}), NewTasks.MABL)
        elif 'MAPS' in task:
            maps_path = './maps_data'
            df = pd.read_excel(f'{maps_path}/test_proverbs.xlsx')

            # Split by subset
            if '/' in task:
                subset = task.split('/')[-1]
                if subset == 'figurative':
                    df = df.loc[df['is_figurative'] == 1, :]
                else:  # non_figurative
                    df = df.loc[df['is_figurative'] == 0, :]

            dset = datasets.Dataset.from_pandas(
                df.rename({
                    'proverb': 'premise', 'conversation': 'context',
                    'answer1': 'choice1', 'answer2': 'choice2', 'answer_key': 'label'
                }, axis='columns')
            )
            cfg_name_to_dset_map[task] = (datasets.DatasetDict({'test': dset}), NewTasks.MAPS)
        elif 'IndoStoryCloze' in task:
            df = datasets.load_dataset('indolem/indo_story_cloze')['test'].to_pandas()

            # Preprocess
            df['premise'] = df.apply(lambda x: '. '.join([
                x['sentence-1'], x['sentence-2'], x['sentence-3'], x['sentence-4']
            ]), axis='columns')
            df = df.rename({'correct_ending': 'choice1', 'incorrect_ending': 'choice2'}, axis='columns')
            df = df[['premise', 'choice1', 'choice2']]
            df['label'] = 0

            dset = datasets.Dataset.from_pandas(df)
            cfg_name_to_dset_map[task] = (datasets.DatasetDict({'test': dset}), NewTasks.IndoStoryCloze)
        elif 'IndoMMLU' in task:
            df = pd.read_csv('indommlu_data/IndoMMLU.csv')
            dset = datasets.Dataset.from_pandas(df.rename({'kunci': 'label'}, axis='columns'))
            cfg_name_to_dset_map[task] = (datasets.DatasetDict({'test': dset}), NewTasks.IndoMMLU)
    return cfg_name_to_dset_map


def load_nlg_datasets():
    nc_conhelp = SEACrowdConfigHelper()
    cfg_name_to_dset_map = {}

    for config_name in NLG_TASK_LIST:
        schema = config_name.split('_')[-1]
        con = nc_conhelp.for_config_name(config_name)
        cfg_name_to_dset_map[config_name] = (con.load_dataset(), list(con.tasks)[0])
    return cfg_name_to_dset_map


### Forget about this for now
def load_flores_datasets():
    dset_map = {}
    for task in FLORES200_TASK_LIST:
        subset = task.replace('flores200-', '')
        src_lang, tgt_lang = subset.split('-')
        dset = datasets.load_dataset('facebook/flores', subset)
        dset = dset.rename_columns({f'sentence_{src_lang}': 'text_1', f'sentence_{tgt_lang}': 'text_2'}).select_columns(
            ['id', 'text_1', 'text_2'])
        dset_map[task] = (dset, Tasks.MACHINE_TRANSLATION)
    return dset_map


### Forget about this for now
def load_truthfulqa_datasets():
    class NewTasks(Enum):
        TRUTHFULQA = "TRUTHFULQA"

    return {'truthfulqa': (datasets.load_from_disk('./truthfulqa_ind'), NewTasks.TRUTHFULQA)}


def load_speech_datasets():
    nc_conhelp = SEACrowdConfigHelper()
    cfg_name_to_dset_map = {}

    for config_name in SPEECH_TASK_LIST:
        print(config_name)
        schema = config_name.split('_')[-1]
        con = nc_conhelp.for_config_name(config_name)
        cfg_name_to_dset_map[config_name] = (con.load_dataset(), list(con.tasks)[0])

    return cfg_name_to_dset_map
