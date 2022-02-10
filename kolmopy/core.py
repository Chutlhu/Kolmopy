import os
from pathlib import Path

class Dataset(object):
    """Kolmopy Dataset Class

    Attributes:
            data_home (str): path where kolmopy will look for the dataset
            name (str): the identifier of the dataset
            bibtex (str or None): dataset citation/s in bibtex format
            remotes (dict or None): data to be downloaded
            readme (str): information about the dataset
    """

    def __init__(self,
        data_home:str=None,
        name:str=None,
        bibtex:str=None,
        remotes:dict=None,
        license:str=None,
        ):
        
        self.name = name
        self.data_home = self.default_path if data_home is None else data_home
        self.bibtex = bibtex
        self.license = license
        self.remotes = remotes

        
        self.variables = {'xyz':None, 't':None} # coordinates, input variables        
        self.fields = {'uvw':None, 'p':None}    # observed, target, output variables
    
    @property
    def default_path(self):
        """Get the default path for the dataset
        Returns:
            str: Local path to the dataset
        """
        sound_datasets_dir = Path('/','.cache','datasets')
        return os.path.join(sound_datasets_dir, self.name)

    def cite(self):
        """
        Print the reference
        """
        print("========== BibTeX ==========")
        print(self.bibtex)

    def download(self):
        raise NotImplementedError

    # def download(self, partial_download=None, force_overwrite=False, cleanup=False):
    #     """Download data to `save_dir` and optionally print a message.
    #     Args:
    #         partial_download (list or None):
    #             A list of keys of remotes to partially download.
    #             If None, all data is downloaded
    #         force_overwrite (bool):
    #             If True, existing files are overwritten by the downloaded files.
    #         cleanup (bool):
    #             Whether to delete any zip/tar files after extracting.
    #     Raises:
    #         ValueError: if invalid keys are passed to partial_download
    #         IOError: if a downloaded file's checksum is different from expected
    #     """
    #     download_utils.downloader(
    #         self.data_home,
    #         remotes=self.remotes,
    #         partial_download=partial_download,
    #         info_message=self._download_info,
    #         force_overwrite=force_overwrite,
    #         cleanup=cleanup,
    #     )

    # def validate(self, verbose=True):
    #     """Validate if the stored dataset is a valid version
    #     Args:
    #         verbose (bool): If False, don't print output
    #     Returns:
    #         * list - files in the index but are missing locally
    #         * list - files which have an invalid checksum
    #     """
    #     missing_files, invalid_checksums = validate.validator(
    #         self._index, self.data_home, verbose=verbose
    #     )
    #     return missing_files, invalid_checksums