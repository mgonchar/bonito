"""
Bonito MKR Utils
"""

from glob import glob
from uuid import UUID
from pathlib import Path
from datetime import timedelta

import numpy as np
import bonito.reader
from tqdm import tqdm
from dateutil import parser
from mkr_format import open_combined_file


class Read(bonito.reader.Read):

    def __init__(self, read, filename, meta=False):

        self.meta = meta

        self.read_id = read.read_id
        self.run_info = read.run_info
        self.filename = filename.name

        self.sample_id = self.run_info.sample_id
        self.run_id = self.run_info.acquisition_id
        self.exp_start_time = self.run_info.acquisition_start_time.isoformat().replace('Z', '')

        self.flow_cell_id = self.run_info.flow_cell_id
        self.device_id = self.run_info.sequencer_position

        if self.meta:
            return

        self.pore = read.pore
        self.mux = self.pore.well
        self.channel = self.pore.channel
        self.read_number = read.read_number
        self.num_samples = read.sample_count
        
        self.context_tags = dict(self.run_info.context_tags)
        self.sample_rate = int(self.context_tags['sample_frequency'])

        self.start = read.start_sample / self.sample_rate
        self.duration = self.num_samples / self.sample_rate

        exp_start_dt = parser.parse(self.exp_start_time)
        start_time = exp_start_dt + timedelta(seconds=self.start)
        self.start_time = start_time.replace(microsecond=0).isoformat()

        self.raw = read.signal

        self.calibration = read.calibration
        self.scaling = self.calibration.scale
        self.offset = self.calibration.offset

        scaled = self.scaling * (self.raw.astype(np.float32) + self.offset)
        trim_start, _ = bonito.reader.trim(scaled[:8000])
        scaled = scaled[trim_start:]
        self.trimmed_samples = trim_start

        self.template_start = self.start + (trim_start / self.sample_rate)
        self.template_duration = self.duration - (trim_start / self.sample_rate)

        self.signal = scaled

        if len(scaled) > 8000:
            med, mad = bonito.reader.med_mad(scaled)
            self.signal = (scaled - med) / max(1.0, mad)
        else:
            self.signal = bonito.reader.norm_by_noisiest_section(scaled)


def mkr_reads(mkr_file, read_ids, skip=False):
    """
    Get all the reads from the `mkr_file`.
    """
    if read_ids is None:
        yield from open_combined_file(mkr_file).reads()
    elif skip:
        for read in open_combined_file(mkr_file).reads():
            if read.read_id not in read_ids:
                yield read
    else:
        yield from open_combined_file(mkr_file).select_reads({UUID(rid) for rid in read_ids}, missing_ok=True)


def get_read_groups(directory, model, read_ids=None, skip=False, n_proc=1, recursive=False, cancel=None):
    """
    Get all the read meta data for a given `directory`.
    """
    groups = set()
    pattern = "**/*.mkr" if recursive else "*.mkr"
    mkr_files = (Path(x) for x in glob(directory + "/" + pattern, recursive=True))

    for mkr_file in mkr_files:
        for read in tqdm(
            mkr_reads(mkr_file, read_ids, skip),
            leave=False, desc="> preprocessing reads", unit=" reads/s", ascii=True, ncols=100
        ):
            read = Read(read, mkr_file, meta=True)
            groups.add(read.readgroup(model))
    return groups


def get_reads(directory, read_ids=None, skip=False, n_proc=1, recursive=False, cancel=None):
    """
    Get all reads in a given `directory`.
    """
    pattern = "**/*.mkr" if recursive else "*.mkr"
    mkr_files = (Path(x) for x in glob(directory + "/" + pattern, recursive=True))

    for mkr_file in mkr_files:
        for read in mkr_reads(mkr_file, read_ids, skip):
            yield Read(read, mkr_file)
            if cancel is not None and cancel.is_set():
                return
