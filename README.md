# Accelerated PIV (Particle Image Velocimetry) in C++ with Python Bindings

This package provides accelerated Particle Image Velocimetry (PIV) functionality implemented in C++ and CUDA with Python bindings.  
PIV is a widely used technique for measuring velocity fields in fluid flows by tracking the motion of particles suspended in the flow.

The project is currently in progress and may be unstable :(

## Dependencies

This project relies on the following dependencies:

- **NVIDIA Cuda Toolkit**
- **OpenCV**
- **C++20**
- **Python3-dev package**
- **Linux (temporary)**

## Installation

You can compile and install the package using CMake and pip:

```bash
git clone https://github.com/AsphodelRem/accelerated_piv_2d.git
cd accelerated_piv_2d
cmake build
pip install .
```

## Usage

```python
import accelerated_piv_cpp

# Define PIV parameters
piv_params = accelerated_piv_cpp.PIVParameters(width=512, height=512, channels=3, window_size=32)

# Create an ImageContainer
image_paths = ["path/to/image1.jpg", "path/to/image2.jpg"]
image_container = accelerated_piv_cpp.ImageContainer(image_paths, piv_params)

# Perform PIV analysis
accelerated_piv_cpp.start_piv_2d(image_container, piv_params)

# Retrieve results
# TODO: Add instructions for retrieving results once available
```

## TO-DO

| # | Task                                                                                        | Done |
|---|---------------------------------------------------------------------------------------------|------|
| 1 | Replace OpenCV with another library for image and video processing                          | ✗    |
| 2 | Implement iterative algorithms for filtering erroneous vectors                              | ✗    |
| 3 | Add export of results to .csv                                                               | ✗    |
| 4 | Add benchmarks                                                                              | ✗    |
| 5 | Add a wrapper for the results storage class                                                 | ✗    |

## API Reference

### `PIVParameters`

Class for defining PIV analysis parameters.

#### Constructor

```python
PIVParameters(width: int, height: int, channels: int, window_size: int, overlap: int = 0,
               filter_type: FilterType = FilterType.kNoFilter, filter_parameter: float = 0.0,
               interpolation_type: InterpolationType = InterpolationType.kGaussian,
               correction_type: CorrectionType = CorrectionType.kNoCorrection,
               correction_parameter: int = 0, save_on_disk: bool = False,
               capacity: int = 1200)
```

#### Attributes

- `width`: Width of the images.
- `height`: Height of the images.
- `channels`: Number of color channels.
- `window_size`: Size of the interrogation window.
- `overlap`: Percentage of overlap between interrogation windows.
- `filter_type`: Type of filter to apply.
- `filter_parameter`: Filter parameter (e.g., cutoff frequency).
- `interpolation_type`: Type of interpolation method.
- `correction_type`: Type of correction method.
- `correction_parameter`: Correction parameter (depends on correction type).
- `save_on_disk`: Whether to save intermediate results on disk.
- `capacity`: Capacity of the data container.

### `ImageContainer`

The ImageContainer serves as a data container for processing images.
It is utilized for loading and pre-processing a set of images intended for PIV analysis.

#### Constructor

```python
ImageContainer(image_paths: List[str], piv_parameters: PIVParameters)
```

### `VideoContainer`

The VideoContainer acts as a data container for processing video files.
It is employed for loading and pre-processing video footage intended for PIV analysis.

#### Constructor

```python
VideoContainer(video_path: str, piv_parameters: PIVParameters)
```

### `start_piv_2d`

Function to start 2D PIV analysis.

```python
start_piv_2d(container: IDataContainer, parameters: PIVParameters)
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
