from sample_patches import run_sample_patches
from generate_node_features import run_generate_node_features
from train import train_run
from predict import run_output_predictions


if __name__ == '__main__':
    ################################################################################
    ##  Define the macros. Change the variables below according to what you need  ##
    ################################################################################

    # Define the unique ID for this run of experiment
    # (i.e. the unique name of the model trained in this experiment)
    run_id = 'demo'

    # Random seed. Change this value if your model failed to converge
    seed = 1024

    # Specify the genome assemblies (reference genome) of the source and target Hi-C/ChIA-PET
    # In this demo, source and target cell lines are sequenced with different assembly genomes
    source_assembly = 'hg19'
    target_assembly = 'hg38'

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

    # Specify the directory that contains the images and graphs of the source cell line
    # You can use a mixed downsampling rate, but here we use an identical sequencing depth
    # for both graph and image
    source_image_data_dir = 'data/txt_gm12878_50'
    source_graph_data_dir = 'data/txt_gm12878_50'

    # Specify the data dir for target cell line
    target_image_data_dir = 'data/txt_hela_100'
    target_graph_data_dir = 'data/txt_hela_100'

    # Name the sampled datasets with unique identifiers you like
    source_dataset_name = 'gm12878_50'
    target_dataset_name = 'hela_100'

    # Define the chromosomes we draw training data from
    source_chroms = [str(i) for i in range(1, 23)] + ['X']
    # Define the chromosomes of the target cell line we want to predict on
    target_chroms = \
        [str(i) for i in range(1, 18)] + \
        [str(i) for i in range(19, 23)] + ['X'] # Chr18 of HeLa-S3 is absent in the Hi-C file

    # Set the threshold cutting off the probability map to generate the final annotations
    threshold = 0.48

    # Set the path to the output file, where saves the annotations
    output_path = 'predictions/demo.bedpe'


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
