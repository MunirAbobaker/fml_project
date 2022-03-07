activate:
		conda activate fml_project

run:
	python main.py play

myagent:
	python main.py play --my-agent my_agent

peaceful:
	python main.py play --agents my_agent random_agent peaceful_agent


train:
	python main.py play --my-agent my_agent --train 1

task_1:
	python main.py play --no-gui --agents my_agent --train 1 --scenario coin-heaven

task_1_with_gui:
	python main.py play --agents my_agent --train 1 --scenario coin-heaven


#task_2:
 	#python main.py play --no-gui --agents my_agent --train 1 --scenario classic

#task_3:
		python main.py play --no-gui --agents my_agent peaceful_agent coin_collector_agent --train 1 --scenario classic

#task4:
 	#python main.py play --no-gui --agents my_agent rule_based_agent --train 1 --scenario classic
