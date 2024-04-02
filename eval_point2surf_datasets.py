import os
import argparse

import eval.eval_point2surf.evaluation as evaluation



parser = argparse.ArgumentParser(description='Evaluate mesh algorithms.')
parser.add_argument("--gendir", type=str,default='results_old/ABC_3k_FKAConv_SA_InterpAttentionKHeadsNet_None/gen_ABCTest_test_3000', help="Path to generated data")
parser.add_argument("--meshdir", type=str, default="meshes/04_pts/")
parser.add_argument("--gtdir", type=str,default='new_datasets/3d_shapes_abc/abc/', help="Path to gt meshes")
parser.add_argument("--workers", type=int, default=4)
args = parser.parse_args()

if __name__ == '__main__':
    new_meshes_dir_abs = os.path.join(args.gendir, args.meshdir)
    ref_meshes_dir_abs = os.path.join(args.gtdir, '03_meshes')
    csv_file = os.path.join(args.gendir, 'hausdorff_dist_pred_rec.csv')
    evaluation.mesh_comparison(
        new_meshes_dir_abs=new_meshes_dir_abs,
        ref_meshes_dir_abs=ref_meshes_dir_abs,
        num_processes=args.workers,
        report_name=csv_file,
        samples_per_model=3000,
        #dataset_file_abs=os.path.join(opt.indir, opt.dataset)
        )