import cooler
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from fanc.commands.fanc_commands import dump
import os
import multiprocessing as mp
from shutil import copyfile


def extract_oe_normalized(clr_path, chroms, txt_dir):
    os.makedirs(txt_dir, exist_ok=True)
    for c in chroms:
        fanc_txt_path = f"{txt_dir}/chr{c}.contact_fanc.txt"
        final_txt_path = f'{txt_dir}/chr{c}.contact.txt'
        command = f'fanc dump -s {c}--{c} -e {clr_path} {fanc_txt_path}'.split(' ')
        dump(command)
        df = pd.read_csv(fanc_txt_path, sep='\t', header=None, dtype={0: 'str', 3: 'str'})
        df[1] = df[1].values - 1
        df[4] = df[4].values - 1
        converted_df = pd.DataFrame({'locus1': df[1], 'locus2': df[4], 'weight': df[6]})
        converted_df.to_csv(
            path_or_buf=final_txt_path,
            header=False, sep='\t', index=False
        )


def normalize_cooler(clr):
    # print('Normalizing {}...'.format(clr.info['cell-type']))
    # with mp.Pool(workers) as pool:
    cooler.balance_cooler(clr, cis_only=True, store=True)
    return clr


def save_to_cool(cool_path, pixel_df, bin_df):
    sanitizer = cooler.create.sanitize_pixels(bin_df, sort=True, tril_action='raise')
    pixel_df = sanitizer(pixel_df.reset_index(drop=True))
    cooler.create_cooler(
        cool_path, bin_df, pixel_df, ordered=True,
        symmetric_upper=True, mode='w', triucheck=True, ensure_sorted=True
    )


def downsampled_chrom_pixels_generator(clr, chroms, rate, bins):
    sanitizer = cooler.create.sanitize_pixels(bins, sort=True, tril_action='raise')
    for chrom in tqdm(chroms):
        pixels = clr.pixels().fetch(chrom)
        v = np.vectorize(lambda x: np.random.binomial(size=1, n=x, p=rate)[0])
        pixels['count'] = v(pixels['count'])
        pixels = pixels[pixels['count'] != 0]
        pixels = pixels.reset_index(drop=True)
        pixels = sanitizer(pixels)
        yield pixels


# Only downsample for cis-matrices
def downsample(clr, rate, out_file, chroms):
    assert rate > 0
    assert rate < 1
    # chroms = clr.chromnames
    # chroms = ['1']
    bins = clr.bins()[:]
    cooler.create_cooler(
        out_file, bins, downsampled_chrom_pixels_generator(clr, chroms, rate, bins),
        ordered=True, symmetric_upper=True, mode='a', triucheck=True, ensure_sorted=True
    )


def align_sequencing_depth(source_cool, target_cool, source_cis_reads, target_cis_reads, source_out, target_out, source_chroms, target_chroms):
    if source_cis_reads > target_cis_reads:
        print('Downsampling source Hi-C to comparable sequencing depth as the target cell line...')
        rate = target_cis_reads / source_cis_reads
        downsample(source_cool, rate, source_out, source_chroms)
        copyfile(target_cool.filename, target_out)
        return cooler.Cooler(source_out), target_cool, int(source_cis_reads*rate), target_cis_reads
    else:
        print('Downsampling target Hi-C to comparable sequencing depth as the source cell line...')
        rate = source_cis_reads / target_cis_reads
        downsample(target_cool, rate, target_out, target_chroms)
        copyfile(source_cool.filename, source_out)
        return source_cool, cooler.Cooler(target_out), source_cis_reads, int(target_cis_reads*rate)


def GCN_downsampling(clr, out_file, cis_reads, is_source, chroms, optimal_depth=265000000):
    if cis_reads < optimal_depth:
        print('Cis-reads fewer than the optimal coverage for GCN. Using lazy mode instead.')
        return clr
    else:
        cell_line = 'Source' if is_source else 'Target'
        print('Downsampling {} cell line for GCN experimental optimum.'.format(cell_line))
        rate = optimal_depth/cis_reads
        downsample(clr, rate, out_file, chroms)
        return cooler.Cooler(out_file)


def rename_cooler_chroms(clr):
    bins = clr.bins()[:]
    _chrom = bins.iloc[0]['chrom']
    if _chrom.startswith('chr'):
        rename_dict = {chr: chr[3:] for chr in clr.chromnames}
        cooler.rename_chroms(clr, rename_dict)
        return clr
    else:
        return clr


if __name__ == '__main__':
    # source_clr = cooler.Cooler('data/gm12878_100.cool')
    # target_clr = cooler.Cooler('tmp/test.mcool::/resolutions/10000')
    # source_clr, target_clr = align_sequencing_depth(
    #     source_clr, target_clr, 2633841149, 47205641, 'tmp/source_unet.cool', 'tmp/target_unet.cool'
    # )
    # normalize_cooler(cooler.Cooler('tmp/source_unet.cool'))
    # print(cooler.Cooler('tmp/source_unet.cool').bins().fetch('1'))
    extract_oe_normalized('tmp/source_unet.cool', ['1'], 'tmp/txt_data')
