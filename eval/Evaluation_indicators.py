import os
import numpy as np
import cv2
from StructureMeasure import structure_measure
from Fmeasure_calu import fmeasure_calu
from Enhancedmeasure import emeasure
from CalMAE import cal_mae
from eval.S_object import s_object
from eval.original_WFb import weighted_f_measure


def evaluate_polyp_segmentation():
    # ---- 1. Model and Dataset_Polyp Settings ----
    models = ['PraNet(original)', 'UNet', 'UNetPP']
    datasets = ['Kvasir', 'CVC-300', 'CVC-ClinicDB', 'ETIS-LaribPolypDB', 'CVC-ColonDB-ColonDB(226-263)']
    variants = ['', '(cycle)']

    base_result_path = r'E:\python_procedure\PraNet-master\results'
    base_gt_path = r'E:\python_procedure\PraNet-master\Dataset'
    base_eval_path = r'E:\python_procedure\PraNet-master\EvaluateResults'

    thresholds = np.linspace(1, 0, 256)

    def calculate_metrics(pred, gt):
        if pred.max() != pred.min():
            pred = (pred - pred.min()) / (pred.max() - pred.min())

        gt_size = gt.shape
        precision, recall, specificity, dice, fmeasure, iou = fmeasure_calu(
            smap=pred,
            gt_map=gt,
            gt_size=gt_size,
            threshold=0.5
        )

        return precision, recall, specificity, dice, iou


    for dataset in datasets:

        gt_path = os.path.join(base_gt_path, dataset, f'masks-{dataset}')


        dataset_eval_path = os.path.join(base_eval_path, dataset)
        os.makedirs(dataset_eval_path, exist_ok=True)


        res_txt = os.path.join(dataset_eval_path, 'comparison_result.txt')

        with open(res_txt, 'w') as f:

            for model in models:

                for variant in variants:
                    experiment_name = f'{dataset}{variant}'
                    print(f'Evaluating model: {model} on experiment: {experiment_name}')


                    res_map_path = os.path.join(base_result_path, model, dataset, experiment_name)

                    if not os.path.exists(res_map_path):
                        print(f"Result path doesn't exist: {res_map_path}")
                        continue

                    img_files = [f for f in os.listdir(res_map_path) if f.endswith('.png')]
                    img_num = len(img_files)

                    if img_num == 0:
                        print(f"No PNG files found in {res_map_path}")
                        continue

                    metrics = {
                        'precision': [], 'recall': [], 'specificity': [],
                        'dice': [], 'iou': [], 'mae': [],
                        'structure_measure': [], 'e_measure': [],
                        'fmeasure': [], 'weighted_fmeasure': [],
                        'sobject': []
                    }

                    for i, img_file in enumerate(img_files):
                        # Load ground truth
                        gt_file = os.path.join(gt_path, img_file)
                        if not os.path.exists(gt_file):
                            print(f"Ground truth file doesn't exist: {gt_file}")
                            continue

                        gt = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE)
                        if gt is None:
                            print(f"Cannot read ground truth image: {gt_file}")
                            continue
                        gt = gt > 128

                        # Load prediction
                        pred_file = os.path.join(res_map_path, img_file)
                        if not os.path.exists(pred_file):
                            print(f"Prediction file doesn't exist: {pred_file}")
                            continue

                        pred = cv2.imread(pred_file, cv2.IMREAD_GRAYSCALE)
                        if pred is None:
                            print(f"Cannot read prediction image: {pred_file}")
                            continue

                        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
                        pred = pred / 255.0

                        # Calculate S-object score
                        so = s_object(pred, gt)
                        metrics['sobject'].append(so)

                        # Calculate weighted F-measure
                        wfm = weighted_f_measure(pred, gt)
                        metrics['weighted_fmeasure'].append(wfm)

                        # Calculate metrics for each threshold
                        for threshold in thresholds:
                            prec, rec, spec, dice, fm, iou = fmeasure_calu(
                                smap=pred,
                                gt_map=gt,
                                gt_size=gt.shape,
                                threshold=threshold
                            )

                            metrics['precision'].append(prec)
                            metrics['recall'].append(rec)
                            metrics['specificity'].append(spec)
                            metrics['dice'].append(dice)
                            metrics['iou'].append(iou)
                            metrics['fmeasure'].append(fm)

                        # Calculate other metrics
                        mae = cal_mae(smap=pred, gt_img=gt)
                        metrics['mae'].append(mae)

                        sm = structure_measure(prediction=pred.astype(np.float64), gt=gt)
                        metrics['structure_measure'].append(sm)

                        em = emeasure(fm=pred, gt=gt)
                        metrics['e_measure'].append(em)

                    if any(metrics.values()):
                        mean_metrics = {k: np.mean(v) for k, v in metrics.items()}

                        result_str = (f'Model: {model}\n'
                                      f'Experiment: {experiment_name}\n'
                                      f'Average Dice: {mean_metrics["dice"]:.4f}\n'
                                      f'Average IoU: {mean_metrics["iou"]:.4f}\n'
                                      f'Average Sensitivity (Recall): {mean_metrics["recall"]:.4f}\n'
                                      f'Average Specificity: {mean_metrics["specificity"]:.4f}\n'
                                      f'Average F-measure: {mean_metrics["fmeasure"]:.4f}\n'
                                      f'Average Weighted F-measure: {mean_metrics["weighted_fmeasure"]:.4f}\n'
                                      f'Average S-object: {mean_metrics["sobject"]:.4f}\n'
                                      f'Average MAE: {mean_metrics["mae"]:.4f}\n'
                                      f'Average Structure Measure: {mean_metrics["structure_measure"]:.4f}\n'
                                      f'Average E-Measure: {mean_metrics["e_measure"]:.4f}\n\n')

                        f.write(result_str)
                        print(result_str)
                    else:
                        print(f"No valid evaluation results for model {model} on experiment {experiment_name}")


if __name__ == '__main__':
    evaluate_polyp_segmentation()