# Sherlock

Sherlock is a human-centric computer vision library that provides optimized algorithms for hand tracking and face detection in real-time.

## Tasks
### General
- [ ] Generate documentation with Sphinx.
- [ ] Deploy API to Amazon AWS Lambda with Kinesis Streams using Boto3.

### Infrastructure
- [x] Refactor from functional programming to OOP.
- [ ] Add tests and bench mark with sample pictures.

### Face Detection
- [x] Implement fast face detection with previous frame data cache.
- [ ] Migrate to current infrastructure.
- [ ] Modularize functionality.
- [ ] Create fast haar cascade module.
- [ ] Implement 3D pose estimation with feature tracking and cyclindrical fitting.

### Handtracking
- [ ] Implement open palm hand tracking.
- [ ] Migrate to current infrastructure.
- [ ] Modularize functionality.
- [ ] Implement advanced methods for hand contour filtering with fast haar cascades.
- [ ] Train custom haar cascades for multiple hand positions.
- [ ] Implement advanced methods for hand model fitting.

## Features

* [Handtracking](http://www.callaunchpad.org)
* [3D Face Pose Estimation](http://www.callaunchpad.org)
* [Object Recognition](http://www.callaunchpad.org)
* [Classifying Human Activity in Videos](http://www.callaunchpad.org)

## License

[Apache Software License](https://github.com/callaunchpad/sherlock/blob/master/LICENSE)
