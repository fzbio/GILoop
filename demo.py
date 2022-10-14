import cooler

from sample_patches import run_sample_patches
from generate_node_features import run_generate_node_features
from train import train_run
from predict import run_output_predictions
from cool_handling import extract_oe_normalized, normalize_cooler, align_sequencing_depth, GCN_downsampling, rename_cooler_chroms
from tempfile import TemporaryDirectory
import os


if __name__ == '__main__':
    ################################################################################
    ##  Define the macros. Change the variables below according to what you need  ##
    ################################################################################

    # Define the unique ID for this run of experiment
    # (i.e. the unique name of the model trained in this experiment)
    run_id = 'demo'
    source_cool_path = 'data/gm12878_100.cool'
    target_cool_path = 'data/hela_100.cool'
    source_cis_reads = 2633841149
    target_cis_reads = 1370071315
    lazy = False

    # Random seed. Change this value if your model failed to converge
    seed = 1024

    # Specify the genome assemblies (reference genome) of the source and target Hi-C/ChIA-PET
    # In this demo, source and target cell lines are sequenced with different assembly genomes
    source_assembly = 'hg19'
    target_assembly = 'hg38'

    # Define the chromosomes we sample training data from
    source_chroms = [str(i) for i in range(1, 23)]
    # Define the chromosomes of the target cell line we want to predict on
    target_chroms = [str(i) for i in range(1, 23)]

    # Set the threshold cutting off the probability map to generate the final annotations
    threshold = 0.48

    # Set the path to the output file, where saves the annotations
    output_path = 'predictions/demo.bedpe'

    # The path to the ChIA-PET annotation files
    source_bedpe_path = 'bedpe/gm12878.tang.ctcf-chiapet.hg19.bedpe'

    # In the real-world scenarios, the target cell line typically does not have ChIA-PET labels
    # In that case, please specify an empty target .bedpe file as a placeholder
    # Comment or uncomment the following two lines as needed
    target_bedpe_path = 'bedpe/hela.hg38.bedpe'
    # target_bedpe_path = 'bedpe/placeholder.bedpe'  # Uncomment this in the case where target label is unavailable

    mode = 'test'
    if 'placeholder' in target_bedpe_path:
        mode = 'realworld'



    temp_dir = TemporaryDirectory()
    source_cooler = cooler.Cooler(source_cool_path)
    target_cooler = cooler.Cooler(target_cool_path)
    source_cooler = rename_cooler_chroms(source_cooler)
    target_cooler = rename_cooler_chroms(target_cooler)
    source_cooler, target_cooler, source_reads_left, target_reads_left = align_sequencing_depth(
        source_cooler, target_cooler, source_cis_reads, target_cis_reads,
        os.path.join(temp_dir.name, 'source_image.cool'), os.path.join(temp_dir.name, 'target_image.cool'),
        source_chroms, target_chroms
    )
    print('Normalizing source and target (image) datasets...')
    source_cooler = normalize_cooler(source_cooler)
    target_cooler = normalize_cooler(target_cooler)
    if not lazy:
        source_graph_cooler = GCN_downsampling(
            source_cooler, os.path.join(temp_dir.name, 'source_graph.cool'), source_reads_left, True, source_chroms
        )
        target_graph_cooler = GCN_downsampling(
            target_cooler, os.path.join(temp_dir.name, 'target_graph.cool'), target_reads_left, False, target_chroms
        )
        print('Normalizing source and target (graph) datasets...')
        source_graph_cooler = normalize_cooler(source_graph_cooler)
        target_graph_cooler = normalize_cooler(target_graph_cooler)

    source_image_data_dir = 'data/txt_{}_source_image'.format(run_id)
    target_image_data_dir = 'data/txt_{}_target_image'.format(run_id)
    extract_oe_normalized(os.path.join(temp_dir.name, 'source_image.cool'), source_chroms, source_image_data_dir)
    extract_oe_normalized(os.path.join(temp_dir.name, 'target_image.cool'), target_chroms, target_image_data_dir)

    if not lazy:
        source_graph_data_dir = 'data/txt_{}_source_graph'.format(run_id)
        target_graph_data_dir = 'data/txt_{}_target_graph'.format(run_id)
        extract_oe_normalized(os.path.join(temp_dir.name, 'source_graph.cool'), source_chroms, source_graph_data_dir)
        extract_oe_normalized(os.path.join(temp_dir.name, 'target_graph.cool'), target_chroms, target_graph_data_dir)
    else:
        source_graph_data_dir = source_image_data_dir
        target_graph_data_dir = target_image_data_dir

    source_dataset_name = '{}_source'.format(run_id)
    target_dataset_name = '{}_target'.format(run_id)


    ##############################################################################
    ###               The GILoop core algorithm starts from here               ###
    ##############################################################################

    # Sample patches for source cell line
    run_sample_patches(
        source_dataset_name,
        source_assembly,
        source_bedpe_path,
        source_image_data_dir,
        source_graph_data_dir,
        source_chroms)
    # Generate the node features for source cell line
    run_generate_node_features(source_dataset_name, source_chroms, source_assembly)

    # Sample patches for target cell line
    run_sample_patches(
        target_dataset_name,
        target_assembly,
        target_bedpe_path,
        target_image_data_dir,
        target_graph_data_dir,
        target_chroms)
    # Generate the node features for target cell line
    run_generate_node_features(target_dataset_name, target_chroms, target_assembly)

    # Train
    train_run(source_chroms, run_id, seed, source_dataset_name, epoch=50)

    # Predict on the target cell line
    run_output_predictions(
        run_id,
        'Finetune',
        threshold,
        target_dataset_name,
        target_assembly,
        target_chroms,
        output_path,
        mode
    )
    temp_dir.cleanup()







