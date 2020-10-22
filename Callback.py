"""
Lucas Hornung --- Quentin Wolak
https://stackoverflow.com/questions/52038651/loss-does-not-decrease-during-training-word2vec-gensim
"""
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models import Word2Vec

# init callback class
class Callback(CallbackAny2Vec):
    """
    Callback to print loss after each epoch
    """
    def __init__(self):
        self.epoch = 0
        self.loss_previous_step = 0
        self.normalised_loss = 0


    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if self.epoch == 0:
            print('Loss after epoch {}: {}'.format(self.epoch, loss/200000))
        else:
            self.normalised_loss = (loss- self.loss_previous_step)/200000
            print('Loss after epoch {}: {}'.format(self.epoch, self.normalised_loss))
        self.epoch += 1
        self.loss_previous_step = loss