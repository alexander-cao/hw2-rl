# hw2-rl
alex cao's homework 2 for rl iems 469 (fall 2020 with prof. klabjan)

# cartpole-v0
policy_value_net.py: policy and value neural networks (each 2 fully connected layers)  
reinforce.py: creates REINFORCE with non-constant baseline loss function (uses Adam optimizer)  
train.py: trains network with minibatch size of 8 episodes per update, plots progress, stops when 100 consecutive episodes with total reward >= 199  

train_output.out: output from training  
train_rewards.png: figure showing mean total reward per training minibatch  
trained_model: saved, trained model  

test_policy.py: tests saved, trained model for 1,000 new episodes and plots progress  
test_output.out: output from testing   
test_rewards.png: figure showing mean total reward per testing episode (perfectly solved)  

# pong-v0
policy_value_net.py: policy and value neural networks (each 1 fully connected layer)  
reinforce.py: creates REINFORCE with non-constant baseline loss function (uses Adam optimizer)  
train.py: trains network with minibatch size of 8 episodes per update, plots progress, saves partially trained model when runninig average of total reward >= -20, -19, etc. i only ran for ~1 day  

train_output.out: output from training  
train_rewards.png: figure showing mean total reward per training minibatch, ran for ~1 day reaching mean reward ~-5  
partially_trained_model: saved, partially trained model 

test_policy.py: tests saved, partially trained model for 1,000 new episodes and plots progress  
test_output.out: output from testing   
test_rewards.png: figure showing mean total reward per testing episode (stopped after 200 episodes for time's sake, mean reward ~-5)  
