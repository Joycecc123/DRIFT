import matplotlib
from batchgenerators.utilities.file_and_folder_operations import join
import os
matplotlib.use('agg')
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
import traceback


class nnUNetLogger(object):
    """
    This class is really trivial. Don't expect cool functionality here. This is my makeshift solution to problems
    arising from out-of-sync epoch numbers and numbers of logged loss values. It also simplifies the trainer class a
    little

    YOU MUST LOG EXACTLY ONE VALUE PER EPOCH FOR EACH OF THE LOGGING ITEMS! DONT FUCK IT UP
    """
    def __init__(self, verbose: bool = False):
        self.my_fantastic_logging = {
            'mean_fg_dice': list(),
            'ema_fg_dice': list(),
            'dice_per_class_or_region': list(),
            'train_losses': list(),
            'val_losses': list(),
            'lrs': list(),
            'epoch_start_timestamps': list(),
            'epoch_end_timestamps': list(),
            'mean_fg_dice_a': list(),
            'mean_fg_dice_b': list(),
            'dice_per_class_a': list(),
            'dice_per_class_b': list(),
            'dice_per_class_combined': list(),
            'weighted_mean_dice': list(),
            'ce_loss': list(),
            'dice_loss': list(),
            'inverse_loss_tumor': list(),
            'inverse_loss_stroma': list(),
            'train_l': list(),
            'train_dc_loss': list(),
            'train_ce_loss': list(),
            'train_inverse_loss': list()
        }
        
        self.verbose = verbose
        # shut up, this logging is great
    def convert_to_python_type(self, value):
        
        import numpy as np
        
       
        if value is None:
            return None
            
       
        if isinstance(value, (list, tuple)):
            return [self.convert_to_python_type(item) for item in value]
            
        
        if isinstance(value, np.ndarray):
            return [self.convert_to_python_type(x) for x in value.tolist()]
            
        
        if isinstance(value, (np.float32, np.float64, np.float16, np.int32, np.int64, np.int16, np.int8)):
            return float(value)
            
        if hasattr(value, 'item'):
            return float(value.item())
            
       
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            return value
        if isinstance(value, bool):
            return value
            
      
        try:
            return float(value)
        except (TypeError, ValueError):
            
            return str(value)

    def log(self, key, value, epoch: int):
        """
        sometimes shit gets messed up. We try to catch that here
        """
        
        value = self.convert_to_python_type(value)
        assert key in self.my_fantastic_logging.keys() and isinstance(self.my_fantastic_logging[key], list), \
            'This function is only intended to log stuff to lists and to have one entry per epoch'

        if self.verbose: print(f'logging {key}: {value} for epoch {epoch}')

        if len(self.my_fantastic_logging[key]) < (epoch + 1):
            self.my_fantastic_logging[key].append(value)
        else:
            assert len(self.my_fantastic_logging[key]) == (epoch + 1), 'something went horribly wrong. My logging ' \
                                                                       'lists length is off by more than 1'
            print(f'maybe some logging issue!? logging {key} and {value}')
            self.my_fantastic_logging[key][epoch] = value

        # handle the ema_fg_dice special case! It is automatically logged when we add a new mean_fg_dice
        if key == 'mean_fg_dice':
            new_ema_pseudo_dice = self.my_fantastic_logging['ema_fg_dice'][epoch - 1] * 0.9 + 0.1 * value \
                if len(self.my_fantastic_logging['ema_fg_dice']) > 0 else value
            self.log('ema_fg_dice', new_ema_pseudo_dice, epoch)
    ##——————————————————————————————————————————————————————————————————————##
    ##————————————————————————one Val———————————————————————————————————————##
    # def plot_progress_png(self, output_folder):
    #     # we infer the epoch form our internal logging
    #     epoch = min([len(i) for i in self.my_fantastic_logging.values()]) - 1  # lists of epoch 0 have len 1
    #     sns.set(font_scale=2.5)
    #     fig, ax_all = plt.subplots(3, 1, figsize=(30, 54))
    #     # regular progress.png as we are used to from previous nnU-Net versions
    #     ax = ax_all[0]
    #     ax2 = ax.twinx()
    #     x_values = list(range(epoch + 1))
    #     ax.plot(x_values, self.my_fantastic_logging['train_losses'][:epoch + 1], color='b', ls='-', label="loss_tr", linewidth=4)
    #     ax.plot(x_values, self.my_fantastic_logging['val_losses'][:epoch + 1], color='r', ls='-', label="loss_val", linewidth=4)
    #     ax2.plot(x_values, self.my_fantastic_logging['mean_fg_dice'][:epoch + 1], color='g', ls='dotted', label="pseudo dice",
    #              linewidth=3)
    #     ax2.plot(x_values, self.my_fantastic_logging['ema_fg_dice'][:epoch + 1], color='g', ls='-', label="pseudo dice (mov. avg.)",
    #              linewidth=4)
    #     ax.set_xlabel("epoch")
    #     ax.set_ylabel("loss")
    #     ax2.set_ylabel("pseudo dice")
    #     ax.legend(loc=(0, 1))
    #     ax2.legend(loc=(0.2, 1))

    #     # epoch times to see whether the training speed is consistent (inconsistent means there are other jobs
    #     # clogging up the system)
    #     ax = ax_all[1]
    #     ax.plot(x_values, [i - j for i, j in zip(self.my_fantastic_logging['epoch_end_timestamps'][:epoch + 1],
    #                                              self.my_fantastic_logging['epoch_start_timestamps'])][:epoch + 1], color='b',
    #             ls='-', label="epoch duration", linewidth=4)
    #     ylim = [0] + [ax.get_ylim()[1]]
    #     ax.set(ylim=ylim)
    #     ax.set_xlabel("epoch")
    #     ax.set_ylabel("time [s]")
    #     ax.legend(loc=(0, 1))

    #     # learning rate
    #     ax = ax_all[2]
    #     ax.plot(x_values, self.my_fantastic_logging['lrs'][:epoch + 1], color='b', ls='-', label="learning rate", linewidth=4)
    #     ax.set_xlabel("epoch")
    #     ax.set_ylabel("learning rate")
    #     ax.legend(loc=(0, 1))

    #     plt.tight_layout()

    #     fig.savefig(join(output_folder, "progress.png"))
    #     plt.close()
    ##——————————————————————————————————————————————————————————————————————##
    ##————————————————————————two val———————————————————————————————————————##
    def plot_progress_png(self, output_folder):
        mpl.use('Agg')
        
        
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
        
        print("\n=== start drawing progress figure===")
        
        
        os.makedirs(output_folder, exist_ok=True)
        if not os.access(output_folder, os.W_OK):
            print(f"error {output_folder} ")
            return
            
        try:
            
            valid_data = {
                k: v for k, v in self.my_fantastic_logging.items() 
                if isinstance(v, list) and len(v) > 0
            }
            
            if not valid_data:
                print("Error: No valid data available for plotting.")
                return
                
            
            plt.clf()
            plt.close('all')
            fig = plt.figure(figsize=(16, 50))
            plt.subplots_adjust(hspace=0.4)  
            
            # 1. training loss figure
            ax1 = plt.subplot(611)

            #print(valid_data)
            if 'train_l' in valid_data:
                ax1.plot(valid_data['train_l'], 'b-', label='Training Loss')
                ax1.set_title('Training Loss', fontsize=14, pad=20)
                ax1.set_xlabel('Epoch', fontsize=12)
                ax1.set_ylabel('Loss', fontsize=12)
                ax1.grid(True, linestyle='--', alpha=0.7)
                ax1.legend(prop={'size': 12, 'family': 'DejaVu Sans'})
                ax1.tick_params(labelsize=10)
    
            ax5 = plt.subplot(614)
            ax5.plot(valid_data['train_dc_loss'], 'b-', label='Training DC Loss')
            ax5.set_title('Training DC Loss', fontsize=14, pad=20)
            ax5.set_xlabel('Epoch', fontsize=12)
            ax5.set_ylabel('Loss', fontsize=12)
            ax5.grid(True, linestyle='--', alpha=0.7)
            ax5.legend(prop={'size': 12, 'family': 'DejaVu Sans'})
            ax5.tick_params(labelsize=10)

            ax6 = plt.subplot(615)
            ax6.plot(valid_data['train_ce_loss'], 'r-', label='Training CE Loss')
            ax6.set_title('Training CE Loss', fontsize=14, pad=20)
            ax6.set_xlabel('Epoch', fontsize=12)
            ax6.set_ylabel('Loss', fontsize=12)
            ax6.grid(True, linestyle='--', alpha=0.7)
            ax6.legend(prop={'size': 12, 'family': 'DejaVu Sans'})
            ax6.tick_params(labelsize=10)

            ax7 = plt.subplot(616)
            ax7.plot(valid_data['train_inverse_loss'], 'g-', label='Training Inverse Loss')
            ax7.set_title('Training Inverse Loss', fontsize=14, pad=20)
            ax7.set_xlabel('Epoch', fontsize=12)
            ax7.set_ylabel('Loss', fontsize=12)
            ax7.grid(True, linestyle='--', alpha=0.7)
            ax7.legend(prop={'size': 12, 'family': 'DejaVu Sans'})
            ax7.tick_params(labelsize=10)
        
        
        
            # 2. dice score figure
            ax2 = plt.subplot(612)
            metrics = [
                ('mean_fg_dice_a', 'Dice Score Group A', 'r-'),
                ('mean_fg_dice_b', 'Dice Score Group B', 'g-'),
                ('weighted_mean_dice', 'Combined Dice Score', 'b-')
            ]
            
            for metric, label, color in metrics:
                if metric in valid_data:
                    ax2.plot(valid_data[metric], color, label=label)
            
            if any(metric[0] in valid_data for metric in metrics):
                ax2.set_title('Dice Scores', fontsize=14, pad=20)
                ax2.set_xlabel('Epoch', fontsize=12)
                ax2.set_ylabel('Dice Score', fontsize=12)
                ax2.grid(True, linestyle='--', alpha=0.7)
                ax2.legend(prop={'size': 12, 'family': 'DejaVu Sans'}, loc='lower right')
                ax2.tick_params(labelsize=10)
            
            # 3. learning rate
            ax3 = plt.subplot(613)
            if 'lrs' in valid_data:
                ax3.plot(valid_data['lrs'], 'g-', label='Learning Rate')
                ax3.set_title('Learning Rate', fontsize=14, pad=20)
                ax3.set_xlabel('Epoch', fontsize=12)
                ax3.set_ylabel('Learning Rate', fontsize=12)
                ax3.grid(True, linestyle='--', alpha=0.7)
                ax3.legend(prop={'size': 12, 'family': 'DejaVu Sans'})
                ax3.tick_params(labelsize=10)
            save_path = os.path.join(output_folder, "progress.png")
            plt.tight_layout(pad=3.0)  
            plt.savefig(save_path, dpi=300, bbox_inches='tight', format='png')
            plt.close()
            
           
            if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
                print(f"Progress plot successfully saved to: {save_path}")
                print(f"File size: {os.path.getsize(save_path)} bytes")
            else:
                print("Error: The plot file was not created successfully")

        except Exception as e:
            print(f"An error occurred during plotting: {str(e)}")
            traceback.print_exc()

            
        finally:
            plt.close('all')  
    def get_checkpoint(self):
        return self.my_fantastic_logging

    def load_checkpoint(self, checkpoint: dict):
        self.my_fantastic_logging = checkpoint
