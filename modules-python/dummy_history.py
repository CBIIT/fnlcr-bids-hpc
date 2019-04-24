class HistoryDummy:
    def __init__(self, mynum):
        self.history = {'val_loss': [mynum-1], 'val_corr': [mynum+1], 'val_dice_coef': [mynum+2]}