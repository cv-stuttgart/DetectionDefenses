class Paths:
  __conf = {
    # Insert paths/to/local/datasets here
    "sintel_mpi": "datasets/Sintel",
    "sintel_subsplit": "datasets/sintel_splitmsk",
    "flying_chairs": "datasets/FlyingChairs_release",
    "flying_things": "datasets/FlyingThings3D",
    "kitti15": "datasets/KITTI",
    "kitti_raw": "datasets/KITTIRaw",
    "hd1k": "datasets/HD1k",
    "spring": "datasets/Spring",
    "driving": "datasets/driving",
  }

  __splits = {
    # Used for dataloading internally
    "sintel_train": "training",
    "sintel_eval": "test",
    "sintel_sub_train": "sintel_train",
    "sintel_sub_eval": "sintel_eval",
    "kitti_train": "training",
    "kitti_eval": "testing",
    "spring_train": "train",
    "spring_eval": "test"
  }

  @staticmethod
  def config(name):
    return Paths.__conf[name]

  @staticmethod
  def splits(name):
    return Paths.__splits[name]

class Conf:
  __conf = {
    # Change the following variables according to your system setup.
    "useCPU": False,  # affects all .to(device) calls

    # Set to False, if your installation of spatial-correlation-sampler
    "correlationSamplerOnlyCPU": True  # only used for PWCNet
  }

  @staticmethod
  def config(name):
    return Conf.__conf[name]


class ProgBar:
  __settings= {
      "disable": False,
      "format_eval": "{desc:19}:{percentage:3.0f}%|{bar:40}{r_bar}",
      "format_train": "{desc:13}:{percentage:3.0f}%|{bar:40}{r_bar}"
  }

  @staticmethod
  def settings(name):
    return ProgBar.__settings[name]
