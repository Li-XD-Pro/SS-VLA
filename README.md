Our code has been modified based on ALFRED. For basic usage information about ALFRED, please refer to [**ALFRED's official GitHub page**](https://github.com/askforalfred/alfred).


First follow the official tutorial to configure the environment and download the dataset, then train the model using the following commands.

```
python models/train/train_seq2seq.py --data data/json_feat_2.1.0 --model seq2seq_im_mask --dout exp/model:{model},name:pm_and_subgoals_01 --splits data/splits/oct21.json --gpu --batch 8 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1
```

![](media/instr_teaser.png)


