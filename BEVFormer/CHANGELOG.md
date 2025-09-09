# Project Optimization and Refactoring Log

## [v1.1.0] - 2025-09-09

### Added
- **Training Script**: Introduced `train_wildtrack.py` to enable fine-tuning the BEVFormer model on the Wildtrack dataset. This allows for adapting the model to the specific characteristics of the dataset, significantly improving performance.
- **Detailed Logging**: Added a `CHANGELOG.md` to track all major changes, enhancing project transparency and maintainability.

### Changed
- **Camera Parameter Parsing**: Modified `wildtrack_dataset.py` to correctly parse real camera intrinsic and extrinsic parameters from XML files, replacing the previous hardcoded default values. This greatly improves the accuracy of spatial transformations.
- **BEV Feature Extraction**: Corrected the logic in `validate_wildtrack.py` to extract the true BEV feature map from the model's forward pass, ensuring the visualization reflects the actual model output.
- **Visualization**: Enhanced `visualize_bev.py` to overlay ground-truth annotations on the BEV heatmap, providing a clear and intuitive way to assess model performance.
- **Configuration**: Updated `custom_wildtrack.py` with a more robust configuration suitable for both fine-tuning and validation.

### Fixed
- **Dependency Management**: Pinned key dependencies in `requirements.txt` to specific versions to prevent environment-related issues and ensure reproducibility.
- **Error Handling**: Improved error messages in data loading and validation scripts to be more informative, helping users quickly identify and resolve setup issues.
- **Code Robustness**: Refactored various parts of the code to improve robustness and readability.
