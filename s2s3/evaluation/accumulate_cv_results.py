import shutil
from typing import Union, List, Tuple
from copy import deepcopy
from batchgenerators.utilities.file_and_folder_operations import (
    subfiles, 
    join, 
    isdir, 
    maybe_mkdir_p, 
    isfile,
    load_json,
    save_json
)
from batchgenerators.utilities.file_and_folder_operations import load_json, join, isdir, maybe_mkdir_p, subfiles, isfile

from nnunetv2.configuration import default_num_processes
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager


# def accumulate_cv_results(trained_model_folder,
#                           merged_output_folder: str,
#                           folds: Union[List[int], Tuple[int, ...]],
#                           num_processes: int = default_num_processes,
#                           overwrite: bool = True):
#     """
#     There are a lot of things that can get fucked up, so the simplest way to deal with potential problems is to
#     collect the cv results into a separate folder and then evaluate them again. No messing with summary_json files!
#     """

#     if overwrite and isdir(merged_output_folder):
#         shutil.rmtree(merged_output_folder)
#     maybe_mkdir_p(merged_output_folder)

#     dataset_json = load_json(join(trained_model_folder, 'dataset.json'))
#     plans_manager = PlansManager(join(trained_model_folder, 'plans.json'))
#     rw = plans_manager.image_reader_writer_class()
#     shutil.copy(join(trained_model_folder, 'dataset.json'), join(merged_output_folder, 'dataset.json'))
#     shutil.copy(join(trained_model_folder, 'plans.json'), join(merged_output_folder, 'plans.json'))

#     did_we_copy_something = False
#     for f in folds:
#         expected_validation_folder = join(trained_model_folder, f'fold_{f}', 'validation')
#         if not isdir(expected_validation_folder):
#             raise RuntimeError(f"fold {f} of model {trained_model_folder} is missing. Please train it!")
#         predicted_files = subfiles(expected_validation_folder, suffix=dataset_json['file_ending'], join=False)
#         for pf in predicted_files:
#             if overwrite and isfile(join(merged_output_folder, pf)):
#                 raise RuntimeError(f'More than one of your folds has a prediction for case {pf}')
#             if overwrite or not isfile(join(merged_output_folder, pf)):
#                 shutil.copy(join(expected_validation_folder, pf), join(merged_output_folder, pf))
#                 did_we_copy_something = True

#     if did_we_copy_something or not isfile(join(merged_output_folder, 'summary.json')):
#         label_manager = plans_manager.get_label_manager(dataset_json)
#         gt_folder = join(nnUNet_raw, plans_manager.dataset_name, 'labelsTr')
#         if not isdir(gt_folder):
#             gt_folder = join(nnUNet_preprocessed, plans_manager.dataset_name, 'gt_segmentations')
#         compute_metrics_on_folder(gt_folder,
#                                   merged_output_folder,
#                                   join(merged_output_folder, 'summary.json'),
#                                   rw,
#                                   dataset_json['file_ending'],
#                                   label_manager.foreground_regions if label_manager.has_regions else
#                                   label_manager.foreground_labels,
#                                   label_manager.ignore_label,
#                                   num_processes)
def label_or_region_to_key(label_or_region: Union[int, Tuple[int]]):
    return str(label_or_region)

def key_to_label_or_region(key: str):
    try:
        return int(key)
    except ValueError:
        key = key.replace('(', '')
        key = key.replace(')', '')
        split = key.split(',')
        return tuple([int(i) for i in split if len(i) > 0])

def save_summary_json(results: dict, output_file: str):
    """
    json does not support tuples as keys (why does it have to be so shitty) so we need to convert that shit
    ourselves
    """
    results_converted = deepcopy(results)
    # convert keys in mean metrics
    if 'mean' in results_converted:
        results_converted['mean'] = {label_or_region_to_key(k): results['mean'][k] 
                                   for k in results['mean'].keys()}
    # convert metric_per_case
    if 'metric_per_case' in results_converted:
        for i in range(len(results_converted["metric_per_case"])):
            if 'metrics' in results_converted["metric_per_case"][i]:
                results_converted["metric_per_case"][i]['metrics'] = \
                    {label_or_region_to_key(k): results["metric_per_case"][i]['metrics'][k]
                     for k in results["metric_per_case"][i]['metrics'].keys()}
    save_json(results_converted, output_file)

def load_summary_json(filename: str):
    results = load_json(filename)
    if 'mean' in results:
        # convert keys in mean metrics
        results['mean'] = {key_to_label_or_region(k): results['mean'][k] 
                          for k in results['mean'].keys()}
    # convert metric_per_case
    if 'metric_per_case' in results:
        for i in range(len(results["metric_per_case"])):
            if 'metrics' in results["metric_per_case"][i]:
                results["metric_per_case"][i]['metrics'] = \
                    {key_to_label_or_region(k): results["metric_per_case"][i]['metrics'][k]
                     for k in results["metric_per_case"][i]['metrics'].keys()}
    return results
def accumulate_cv_results(trained_model_folder,
                         merged_output_folder: str,
                         folds: Union[List[int], Tuple[int, ...]],
                         num_processes: int = default_num_processes,
                         overwrite: bool = True):
    """
    累积交叉验证结果并分别计算group_a和group_b的指标
    """
    if overwrite and isdir(merged_output_folder):
        shutil.rmtree(merged_output_folder)
    maybe_mkdir_p(merged_output_folder)

    # 复制必要的文件
    dataset_json = load_json(join(trained_model_folder, 'dataset.json'))
    plans_manager = PlansManager(join(trained_model_folder, 'plans.json'))
    rw = plans_manager.image_reader_writer_class()
    shutil.copy(join(trained_model_folder, 'dataset.json'), join(merged_output_folder, 'dataset.json'))
    shutil.copy(join(trained_model_folder, 'plans.json'), join(merged_output_folder, 'plans.json'))

    # 创建group_a和group_b的输出文件夹
    group_a_folder = join(merged_output_folder, 'group_a')
    group_b_folder = join(merged_output_folder, 'group_b')
    maybe_mkdir_p(group_a_folder)
    maybe_mkdir_p(group_b_folder)

    did_we_copy_something = False

    # 处理每个fold
    for f in folds:
        expected_validation_folder = join(trained_model_folder, f'fold_{f}', 'validation')
        if not isdir(expected_validation_folder):
            raise RuntimeError(f"fold {f} of model {trained_model_folder} is missing. Please train it!")
        
        # 处理group_a
        group_a_val_folder = join(expected_validation_folder, 'group_a')
        if isdir(group_a_val_folder):
            predicted_files = subfiles(group_a_val_folder, suffix=dataset_json['file_ending'], join=False)
            for pf in predicted_files:
                # 复制到group_a文件夹，保持原文件名
                shutil.copy(join(group_a_val_folder, pf), join(group_a_folder, pf))
                did_we_copy_something = True

        # 处理group_b
        group_b_val_folder = join(expected_validation_folder, 'group_b')
        if isdir(group_b_val_folder):
            predicted_files = subfiles(group_b_val_folder, suffix=dataset_json['file_ending'], join=False)
            for pf in predicted_files:
                # 复制到group_b文件夹，保持原文件名
                shutil.copy(join(group_b_val_folder, pf), join(group_b_folder, pf))
                did_we_copy_something = True

    # 如果有复制文件，计算指标
    if did_we_copy_something or not isfile(join(merged_output_folder, 'summary.json')):
        label_manager = plans_manager.get_label_manager(dataset_json)
        gt_folder = join(nnUNet_raw, plans_manager.dataset_name, 'labelsTr')
        if not isdir(gt_folder):
            gt_folder = join(nnUNet_preprocessed, plans_manager.dataset_name, 'gt_segmentations')

        # 计算group_a的指标
        print("\nComputing metrics for Group A...")
        metrics_a = compute_metrics_on_folder(
            gt_folder,
            group_a_folder,
            join(group_a_folder, 'summary.json'),
            rw,
            dataset_json['file_ending'],
            label_manager.foreground_regions if label_manager.has_regions else label_manager.foreground_labels,
            label_manager.ignore_label,
            num_processes
        )

        # 计算group_b的指标
        print("\nComputing metrics for Group B...")
        metrics_b = compute_metrics_on_folder(
            gt_folder,
            group_b_folder,
            join(group_b_folder, 'summary.json'),
            rw,
            dataset_json['file_ending'],
            label_manager.foreground_regions if label_manager.has_regions else label_manager.foreground_labels,
            label_manager.ignore_label,
            num_processes
        )

        # 构造完整的summary
        metric_per_case = []
        mean_metrics = {}
        
        # 合并所有case的指标
        if 'metric_per_case' in metrics_a:
            metric_per_case.extend(metrics_a['metric_per_case'])
        if 'metric_per_case' in metrics_b:
            metric_per_case.extend(metrics_b['metric_per_case'])
            
        # 构造mean部分
        region_keys = set()
        if 'mean' in metrics_a:
            region_keys.update(metrics_a['mean'].keys())
        if 'mean' in metrics_b:
            region_keys.update(metrics_b['mean'].keys())
            
        # 计算每个region的平均值
        for region in region_keys:
            mean_metrics[region] = {}
            metrics_a_region = metrics_a['mean'].get(region, {})
            metrics_b_region = metrics_b['mean'].get(region, {})
            
            for metric in ['Dice', 'IoU', 'FP', 'TP', 'FN', 'TN', 'n_pred', 'n_ref']:
                if metric in metrics_a_region and metric in metrics_b_region:
                    mean_metrics[region][metric] = (metrics_a_region[metric] + metrics_b_region[metric]) / 2

        # 计算总体平均值
        mean_dice = (metrics_a['foreground_mean']['Dice'] + metrics_b['foreground_mean']['Dice']) / 2
        
        # 构造最终的summary
        summary = {
            'metric_per_case': metric_per_case,
            'mean': mean_metrics,
            'foreground_mean': {
                'Dice': mean_dice
            },
            'group_results': {
                'group_a': metrics_a['foreground_mean'],
                'group_b': metrics_b['foreground_mean']
            }
        }
        
        # 保存summary
        save_summary_json(summary, join(merged_output_folder, 'summary.json'))

        # 打印结果
        print('\nFinal Results:')
        print(f"Group A Dice: {metrics_a['foreground_mean']['Dice']:.4f}")
        print(f"Group B Dice: {metrics_b['foreground_mean']['Dice']:.4f}")
        print(f"Average Dice: {mean_dice:.4f}")