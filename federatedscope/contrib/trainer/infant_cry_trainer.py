from federatedscope.core.trainers.trainer import GeneralTrainer
from federatedscope.register import register_trainer

class MyTrainer(GeneralTrainer):
    pass

def call_my_trainer(trainer_type):
    if trainer_type == 'mytrainer':
        trainer_builder = MyTrainer
        return trainer_builder

register_trainer('mytrainer', call_my_trainer)
