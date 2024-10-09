# CornNet

The dataset can be found from   https://drive.google.com/file/d/1ptrLdZPKgEUpX2jKBCrHKeIeCZwfhTHC/view?usp=sharing

train_embeddings is used to pretrain KG embeddings.

Different datasets

<blockquote>
human written reformulation\
self.conversation_path = 'data/CONQUER/ConvRef_trainset_processed.json'\
self.conversation_valid_path = 'data/CONQUER/ConvRef_devset_processed.json'\
self.conversation_test_path = 'data/CONQUER/ConvRef_testset_processed.json'
</blockquote>

<blockquote>
bart reformulation \
self.conversation_path = 'bart_ref_wo_ans/ConvRef_trainset_processed.json'\
self.conversation_valid_path = 'bart_ref_wo_ans/ConvRef_devset_processed.json'\
self.conversation_test_path = 'bart_ref_wo_ans/ConvRef_testset_processed.json'
</blockquote>

<blockquote>
GPT2 reformulation\
self.conversation_path = 'gpt2_ref_wo_ans/ConvRef_trainset_processed.json'\
self.conversation_valid_path = 'gpt2_ref_wo_ans/ConvRef_devset_processed.json'\
self.conversation_test_path = 'gpt2_ref_wo_ans/ConvRef_testset_processed.json'
</blockquote>


data directory is the dataset for ConvRef. 

For convex dataset, please directly download from its original github repository

To run the model, simply go to the corresponding directory, and run the main.py. 
Change the paramters in the corresponding config.py should be ok.



If you think it is useful, please consider cite our paper. 

@inproceedings{liu-etal-2024-conversational, \
    title = "Conversational Question Answering with Language Models Generated Reformulations over Knowledge Graph", \
    author = "Liu, Lihui  and 
      Hill, Blaine  and
      Du, Boxin  and
      Wang, Fei  and
      Tong, Hanghang", \
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek", \
    booktitle = "Findings of the Association for Computational Linguistics ACL 2024", \
    month = aug, \
    year = "2024", \
    address = "Bangkok, Thailand and virtual meeting", \
    publisher = "Association for Computational Linguistics", \
}
