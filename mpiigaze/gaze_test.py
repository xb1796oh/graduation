from typing import Optional
import argparse
import datetime
import logging
import pathlib

import cv2
import numpy as np
from ptgaze import (Face, FacePartsName, GazeEstimationMethod, GazeEstimator,
                    Visualizer)
from ptgaze import get_default_config
from ptgaze.utils import update_default_config, update_config
from scipy.spatial import distance
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class gaze_estimate:
    QUIT_KEYS = {27, ord('q')}

    def __init__(self):
        
        self.config = self.get_config()
        self.gaze_estimator = GazeEstimator(self.config)
        self.visualizer = Visualizer(self.gaze_estimator.camera)

        """
        self.cap = self._create_capture()
        self.output_dir = self._create_output_dir()
        self.writer = self._create_video_writer()
        """

        self.stop = False
        self.show_bbox = self.config.demo.show_bbox
        self.show_head_pose = self.config.demo.show_head_pose
        self.show_landmarks = self.config.demo.show_landmarks
        self.show_normalized_image = self.config.demo.show_normalized_image
        self.show_template_model = self.config.demo.show_template_model

        self.space_img = np.full((640, 640, 3),255, dtype = np.uint8)


    def get_config(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--config',
            type=str,
            help='Config file for YACS. When using a config file, all the other '
            'commandline arguments are ignored. '
            'See https://github.com/hysts/pytorch_mpiigaze_demo/configs/demo_mpiigaze.yaml'
        )
        parser.add_argument(
            '--mode',
            type=str,
            default='face',
            choices=['eye', 'face'],
            help='With \'eye\', MPIIGaze model will be used. With \'face\', '
            'MPIIFaceGaze model will be used. (default: \'eye\')')
        parser.add_argument(
            '--face-detector',
            type=str,
            default='dlib',
            choices=['dlib', 'face_alignment_dlib', 'face_alignment_sfd'],
            help='The method used to detect faces and find face landmarks '
            '(default: \'dlib\')')
        parser.add_argument('--device',
                            type=str,
                            choices=['cpu', 'cuda'],
                            help='Device used for model inference.')
        parser.add_argument('--image',
                            type=str,
                            help='Path to an input image file.')
        parser.add_argument('--video',
                            type=str,
                            help='Path to an input video file.')
        parser.add_argument(
            '--camera',
            type=str,
            help='Camera calibration file. '
            'See https://github.com/hysts/pytorch_mpiigaze_demo/ptgaze/data/calib/sample_params.yaml'
        )
        parser.add_argument(
            '--output-dir',
            '-o',
            type=str,
            help='If specified, the overlaid video will be saved to this directory.'
        )
        parser.add_argument('--ext',
                            '-e',
                            type=str,
                            choices=['avi', 'mp4'],
                            help='Output video file extension.')
        parser.add_argument(
            '--no-screen',
            action='store_true',
            help='If specified, the video is not displayed on screen, and saved '
            'to the output directory.')
        parser.add_argument('--debug', action='store_true')
        args = parser.parse_args()

        if args.debug:
            logging.getLogger('ptgaze').setLevel(logging.DEBUG)

        config = get_default_config()
      
        if args.config:
            config.merge_from_file(args.config)
            if (args.device or args.image or args.video or args.camera
                    or args.output_dir or args.ext or args.no_screen):
                raise RuntimeError(
                    'When using a config file, all the other commandline '
                    'arguments are ignored.')
            if config.demo.image_path and config.demo.video_path:
                raise ValueError(
                    'Only one of config.demo.image_path or config.demo.video_path '
                    'can be specified.')
        else:
            update_default_config(config, args)

        update_config(config)   
        return config

    

    def _run_get_image(self, frame):
        
        x,y,z,distance,pitch,yaw = self.get_vec(frame)
        return x,y,z,distance,pitch,yaw


    def get_vec(self, image) -> None:
         
        undistorted = cv2.undistort(
        image, self.gaze_estimator.camera.camera_matrix,
        self.gaze_estimator.camera.dist_coefficients)
        x = 0
        y=0
        z=0
        distance=0
        pitch=0 
        yaw = 0
        self.visualizer.set_image(image.copy())
        faces = self.gaze_estimator.detect_faces(undistorted)
        for face in faces:
            self.gaze_estimator.estimate_gaze(undistorted, face)
            euler_angles = face.head_pose_rot.as_euler('XYZ', degrees=True)
            x,y,z = face.change_coordinate_system(euler_angles)
            distance = face.distance
            pitch, yaw = np.rad2deg(face.vector_to_angle(face.gaze_vector))
           
        if self.config.demo.use_camera:
            self.visualizer.image = self.visualizer.image[:, ::-1]
    
        return x,y,z,distance,pitch,yaw
       




    def run(self) -> None:
        if self.config.demo.use_camera or self.config.demo.video_path:
            self._run_on_video()
        elif self.config.demo.image_path:
            self._run_on_image()
        else:
            raise ValueError


    def _run_on_image(self):
        image = cv2.imread(self.config.demo.image_path)
        self._process_image(image)
        if self.config.demo.display_on_screen:
            while True:
                key_pressed = self._wait_key()
                if self.stop:
                    break
                if key_pressed:
                    self._process_image(image)
                cv2.imshow('image', self.visualizer.image)
        if self.config.demo.output_dir:
            name = pathlib.Path(self.config.demo.image_path).name
            output_path = pathlib.Path(self.config.demo.output_dir) / name
            cv2.imwrite(output_path.as_posix(), self.visualizer.image)

    def _run_on_video(self) -> None:
        while True:
            if self.config.demo.display_on_screen:
                self._wait_key()
                if self.stop:
                    break

            ok, frame = self.cap.read()
            if not ok:
                break
            self._process_image(frame)
            
            if self.config.demo.display_on_screen:
                cv2.imshow('ROI', self.space_img)
                cv2.imshow('frame', self.visualizer.image)
        self.cap.release()
        if self.writer:
            self.writer.release()

    def _process_image(self, image) -> None:
        undistorted = cv2.undistort(
            image, self.gaze_estimator.camera.camera_matrix,
            self.gaze_estimator.camera.dist_coefficients)

        self.visualizer.set_image(image.copy())
        faces = self.gaze_estimator.detect_faces(undistorted)
        for face in faces:
            self.gaze_estimator.estimate_gaze(undistorted, face)
            self._draw_face_bbox(face)
            self._draw_head_pose(face)
            self._draw_landmarks(face)
            self._draw_face_template_model(face)
            self._draw_gaze_vector(face)
            self._display_normalized_image(face)

        if self.config.demo.use_camera:
            self.visualizer.image = self.visualizer.image[:, ::-1]
        if self.writer:
            self.writer.write(self.visualizer.image)

    def _create_capture(self) -> Optional[cv2.VideoCapture]:
        if self.config.demo.image_path:
            return None
        if self.config.demo.use_camera:
            cap = cv2.VideoCapture(0)
        elif self.config.demo.video_path:
            cap = cv2.VideoCapture(self.config.demo.video_path)
        else:
            raise ValueError
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.gaze_estimator.camera.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.gaze_estimator.camera.height)
        return cap

    def _create_output_dir(self) -> Optional[pathlib.Path]:
        if not self.config.demo.output_dir:
            return
        output_dir = pathlib.Path(self.config.demo.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        return output_dir

    @staticmethod
    def _create_timestamp() -> str:
        dt = datetime.datetime.now()
        return dt.strftime('%Y%m%d_%H%M%S')

    def _create_video_writer(self) -> Optional[cv2.VideoWriter]:
        if self.config.demo.image_path:
            return None
        if not self.output_dir:
            return None
        ext = self.config.demo.output_file_extension
        if ext == 'mp4':
            fourcc = cv2.VideoWriter_fourcc(*'H264')
        elif ext == 'avi':
            fourcc = cv2.VideoWriter_fourcc(*'PIM1')
        else:
            raise ValueError
        if self.config.demo.use_camera:
            output_name = f'{self._create_timestamp()}.{ext}'
        elif self.config.demo.video_path:
            name = pathlib.Path(self.config.demo.video_path).stem
            output_name = f'{name}.{ext}'
        else:
            raise ValueError
        output_path = self.output_dir / output_name
        writer = cv2.VideoWriter(output_path.as_posix(), fourcc, 30,
                                 (self.gaze_estimator.camera.width,
                                  self.gaze_estimator.camera.height))
        if writer is None:
            raise RuntimeError
        return writer

    def _wait_key(self) -> bool:
        key = cv2.waitKey(self.config.demo.wait_time) & 0xff
        if key in self.QUIT_KEYS:
            self.stop = True
        elif key == ord('b'):
            self.show_bbox = not self.show_bbox
        elif key == ord('l'):
            self.show_landmarks = not self.show_landmarks
        elif key == ord('h'):
            self.show_head_pose = not self.show_head_pose
        elif key == ord('n'):
            self.show_normalized_image = not self.show_normalized_image
        elif key == ord('t'):
            self.show_template_model = not self.show_template_model
        else:
            return False
        return True

    def _draw_face_bbox(self, face: Face) -> None:
        if not self.show_bbox:
            return
        self.visualizer.draw_bbox(face.bbox)

    def _draw_head_pose(self, face: Face) -> None:
        if not self.show_head_pose:
            return
        # Draw the axes of the model coordinate system
        length = self.config.demo.head_pose_axis_length
        self.visualizer.draw_model_axes(face, length, lw=2)

        euler_angles = face.head_pose_rot.as_euler('XYZ', degrees=True)
        pitch, yaw, roll = face.change_coordinate_system(euler_angles)
        #logger.info(f'[head] pitch: {pitch:.2f}, yaw: {yaw:.2f}, '
        #            f'roll: {roll:.2f}, distance: {face.distance:.2f}')

    def _draw_landmarks(self, face: Face) -> None:
        if not self.show_landmarks:
            return
        self.visualizer.draw_points(face.landmarks,
                                    color=(0, 255, 255),
                                    size=1)

    def _draw_face_template_model(self, face: Face) -> None:
        if not self.show_template_model:
            return
        self.visualizer.draw_3d_points(face.model3d,
                                       color=(255, 0, 525),
                                       size=1)

    def _display_normalized_image(self, face: Face) -> None:
        if not self.config.demo.display_on_screen:
            return
        if not self.show_normalized_image:
            return
        if self.config.mode == GazeEstimationMethod.MPIIGaze.name:
            reye = face.reye.normalized_image
            leye = face.leye.normalized_image
            normalized = np.hstack([reye, leye])
        elif self.config.mode == GazeEstimationMethod.MPIIFaceGaze.name:
            normalized = face.normalized_image
        else:
            raise ValueError
        if self.config.demo.use_camera:
            normalized = normalized[:, ::-1]
        cv2.imshow('normalized', normalized)

    def _draw_gaze_vector(self, face: Face) -> None:
        length = self.config.demo.gaze_visualization_length
        if self.config.mode == GazeEstimationMethod.MPIIGaze.name:
            for key in [FacePartsName.REYE, FacePartsName.LEYE]:
                eye = getattr(face, key.name.lower())
                self.visualizer.draw_3d_line(
                    eye.center, eye.center + length * eye.gaze_vector)
                pitch, yaw = np.rad2deg(eye.vector_to_angle(eye.gaze_vector))
                logger.info(
                    f'[{key.name.lower()}] pitch: {pitch:.2f}, yaw: {yaw:.2f}')

        elif self.config.mode == GazeEstimationMethod.MPIIFaceGaze.name:
            self.visualizer.draw_3d_line(
                face.center, face.center + length * face.gaze_vector)
            pitch, yaw = np.rad2deg(face.vector_to_angle(face.gaze_vector))
            logger.info(f'[face] pitch: {pitch:.2f}, yaw: {yaw:.2f}')
        else:
            raise ValueError

if __name__ == "__main__":
    demo = gaze_estimate()         
    demo.run()   