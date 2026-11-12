# models/

Model weights live here. They are not stored in the repository:

- `yolo26n.pt`, `yolo26s.pt`, ... are downloaded automatically the first time a
  model size is used (`setup.bat` fetches nano and small up front).
- `custom_<project>.pt` files are the models you train yourself in the
  **Train Custom Object** window.

Once a file is here the app uses it without internet.
