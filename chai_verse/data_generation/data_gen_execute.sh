

# Example
python /home/ec2-user/efs-global/chai/ChaiCodingChallenge/chai_verse/data_generation/data_gen.py --example
# Just Rollplay
python /home/ec2-user/efs-global/chai/ChaiCodingChallenge/chai_verse/data_gen.py \
  --out /home/ec2-user/efs-global/chai/gen_dataset/all_roles/p_4/d_1000 \
  --n_prompts 1000

python /home/ec2-user/efs-global/chai/ChaiCodingChallenge/chai_verse/data_generation/data_gen.py --out /home/ec2-user/efs-global/chai/gen_dataset/all_roles/p_1_1000 --n_prompts 1000

#Relationships
python /home/ec2-user/efs-global/chai/ChaiCodingChallenge/chai_verse/data_generation/data_gen.py \
  --out /home/ec2-user/efs-global/chai/gen_dataset/relationship/p_3/d_1000 \
  --n_prompts 1000 \
  --prob_relationship 33 \
  --prob_roleplay 0 \
  --prob_roleplay_relationship 33 \
  --prob_fantasy 0 \
  --prob_mafia 0 \
  --prob_mafia_relationship 33 \
  --prob_mafia_celebrity 0