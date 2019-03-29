python run_classifier_with_early_stoppage.py \
    --data_dir=D:/bert-folder/train_val_test \
    --bert_model=D:/bert-folder/pretrained_models/uncased_L-12_H-768_A-12 \
    --task_name=ste \
    --output_dir=D:/bert-folder/early_stoppage_output/512_2e-5 \
    --gradient_accumulation_steps=21 \
    --train_batch_size=63 \
    --learning_rate=2e-5 \
    --num_train_epochs=12 \
    --max_seq_length=512 \
    --do_lower_case \
    --do_train \
    --do_eval \
    --do_test \
 
python run_classifier_with_early_stoppage.py \
    --data_dir=D:/bert-folder/train_val_test \
    --bert_model=D:/bert-folder/pretrained_models/uncased_L-12_H-768_A-12 \
    --task_name=ste \
    --output_dir=D:/bert-folder/early_stoppage_output/512_3e-5 \
    --gradient_accumulation_steps=21 \
    --train_batch_size=63 \
    --learning_rate=3e-5 \
    --num_train_epochs=12 \
    --max_seq_length=512 \
    --do_lower_case \
    --do_train \
    --do_eval \
    --do_test \
 
python run_classifier_with_early_stoppage.py \
    --data_dir=D:/bert-folder/train_val_test \
    --bert_model=D:/bert-folder/pretrained_models/uncased_L-12_H-768_A-12 \
    --task_name=ste \
    --output_dir=D:/bert-folder/early_stoppage_output/512_4e-5 \
    --gradient_accumulation_steps=21 \
    --train_batch_size=63 \
    --learning_rate=4e-5 \
    --num_train_epochs=12 \
    --max_seq_length=512 \
    --do_lower_case \
    --do_train \
    --do_eval \
    --do_test \
 
python run_classifier_with_early_stoppage.py \
    --data_dir=D:/bert-folder/train_val_test \
    --bert_model=D:/bert-folder/pretrained_models/uncased_L-12_H-768_A-12 \
    --task_name=ste \
    --output_dir=D:/bert-folder/early_stoppage_output/512_5e-5 \
    --gradient_accumulation_steps=21 \
    --train_batch_size=63 \
    --learning_rate=5e-5 \
    --num_train_epochs=12 \
    --max_seq_length=512 \
    --do_lower_case \
    --do_train \
    --do_eval \
    --do_test \

