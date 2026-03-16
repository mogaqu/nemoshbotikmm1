import torch
from ultralytics import YOLO
import cv2
import numpy as np
from typing import Dict, Any, Tuple, List, Optional

# Optional onnx import for reading fixed input shapes of ONNX models
try:
    import onnx
except Exception:
    onnx = None

class DetectionManager:
    """Handles model loading, frame preprocessing, and object detection."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the DetectionManager.

        Args:
            config: The main configuration dictionary for the bot.
        """
        self.config: Dict[str, Any] = config
        # Check if models are ONNX format BEFORE initialization
        self.is_candy_onnx = config['weights']['candy'].lower().endswith('.onnx')
        player_path = config['weights'].get('player')
        self.is_player_onnx = isinstance(player_path, str) and player_path.lower().endswith('.onnx')
        
        self.device: torch.device = self._initialize_device()
        self.model: YOLO
        self.player_model: Optional[YOLO]
        self.player_class_ids: Optional[set]
        self.player_class_labels: List[str]
        
        # Check ONNX CUDA support and adjust device if needed
        self.onnx_cuda_supported = self._check_onnx_cuda_support()
        if not self.onnx_cuda_supported and self.device.type == 'cuda':
            print("INFO: ONNX CUDA not supported, using CPU for ONNX models")
            
        (self.model, self.player_model, self.player_class_ids, self.player_class_labels,
         self.candy_onnx_input_size, self.player_onnx_input_size) = self._initialize_models(
            config['weights']['candy'], config['weights'].get('player')
        )
        
        # For ONNX models, we don't use half precision as they're already optimized
        # Also, ONNX Runtime may not support CUDA, so we'll fallback to CPU
        self.use_half: bool = self.device.type == 'cuda' and self.config['predator_mode']['use_fp16'] and not self.is_candy_onnx
        self.preprocess_enabled: bool = self.config['preprocessing']['enabled']
        self.preprocess_s_gain: float = self.config['preprocessing']['s_gain']
        self.preprocess_v_gain: float = self.config['preprocessing']['v_gain']

    def _initialize_device(self) -> torch.device:
        """
        Initializes the computation device (CUDA or CPU).

        Returns:
            The initialized torch device.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            print("INFO: Using CUDA (GPU) for detection.")
        else:
            print("INFO: CUDA not available, using CPU for detection.")
        return device
    
    def _check_onnx_cuda_support(self) -> bool:
        """
        Checks if CUDA is properly supported for ONNX Runtime.
        User doesn't want CPU fallback, so we return False if CUDA won't work.
        
        Returns:
            True if CUDA works with ONNX, False otherwise.
        """
        # Check if we're using ONNX models at all
        if not (self.is_candy_onnx or self.is_player_onnx):
            return True  # Not using ONNX, no need to check
        
        # If device is CPU, don't bother checking CUDA
        if self.device.type != 'cuda':
            return False
        
        try:
            import onnxruntime as ort
            # Try to get CUDA execution provider info
            available_providers = ort.get_available_providers()
            print(f"INFO: Available ONNX providers: {available_providers}")
            
            if 'CUDAExecutionProvider' in available_providers:
                # User wants GPU, so we need to try CUDA even if it might fail
                # The error messages will help diagnose the issue
                print("INFO: Attempting to use CUDA for ONNX models (user preference: no CPU fallback)")
                return True
            else:
                print("ERROR: CUDAExecutionProvider not available for ONNX models!")
                print("ERROR: Cannot use ONNX models without CUDA (CPU fallback disabled by user)")
                return False
        except Exception as e:
            print(f"ERROR: Could not check ONNX CUDA support: {e}")
            return False

    def _initialize_models(self, weights_path: str, player_weights_path: Optional[str]) -> Tuple[YOLO, Optional[YOLO], Optional[set], List[str]]:
        """
        Loads the YOLO models for candy and player detection.

        Args:
            weights_path: Path to the candy detection model weights.
            player_weights_path: Path to the player detection model weights.

        Returns:
            A tuple containing the loaded models and player class information.
        """
        # Check if models are ONNX format
        is_candy_onnx = weights_path.lower().endswith('.onnx')
        is_player_onnx = player_weights_path and player_weights_path.lower().endswith('.onnx')
        
        # Load candy model
        if is_candy_onnx:
            model = YOLO(weights_path, task='detect')
        else:
            model = YOLO(weights_path).to(self.device)
        
        player_model = None
        player_class_ids = None
        player_class_labels = []

        if player_weights_path:
            # Load player model
            if is_player_onnx:
                player_model = YOLO(player_weights_path, task='detect')
            else:
                player_model = YOLO(player_weights_path).to(self.device)
            try:
                raw_names = getattr(getattr(player_model, 'model', None), 'names', None) or \
                            getattr(player_model, 'names', None)
                
                if isinstance(raw_names, dict):
                    player_class_ids = {int(k) for k in raw_names.keys()}
                    player_class_labels = [str(raw_names[k]) for k in sorted(player_class_ids)]
                elif isinstance(raw_names, (list, tuple)):
                    player_class_ids = set(range(len(raw_names)))
                    player_class_labels = [str(raw_names[idx]) for idx in sorted(player_class_ids)]
                elif raw_names is not None:
                    player_class_ids = {0}
                    player_class_labels = [str(raw_names)]

            except Exception as e:
                print(f"WARN: Could not determine player classes: {e}")
                player_class_ids = {0}
                player_class_labels = []
        
        # Try to detect fixed input sizes for ONNX models (height/width). If unable, return None.
        candy_onnx_input_size = None
        player_onnx_input_size = None

        try:
            if is_candy_onnx and onnx is not None:
                onnx_model = onnx.load(weights_path)
                # typically input shape: [batch, channels, height, width]
                input_shape = onnx_model.graph.input[0].type.tensor_type.shape.dim
                if len(input_shape) >= 4:
                    h = getattr(input_shape[2], 'dim_value', 0)
                    w = getattr(input_shape[3], 'dim_value', 0)
                    if h and w:
                        # if both dims equal, use that; otherwise keep max
                        candy_onnx_input_size = int(max(h, w))
        except Exception:
            candy_onnx_input_size = None

        try:
            if is_player_onnx and onnx is not None and player_weights_path:
                onnx_model = onnx.load(player_weights_path)
                input_shape = onnx_model.graph.input[0].type.tensor_type.shape.dim
                if len(input_shape) >= 4:
                    h = getattr(input_shape[2], 'dim_value', 0)
                    w = getattr(input_shape[3], 'dim_value', 0)
                    if h and w:
                        player_onnx_input_size = int(max(h, w))
        except Exception:
            player_onnx_input_size = None

        return model, player_model, player_class_ids, player_class_labels, candy_onnx_input_size, player_onnx_input_size

    def preprocess_frame(self, img: np.ndarray) -> np.ndarray:
        """
        Applies preprocessing to the input image frame.

        Args:
            img: The input image in BGR format.

        Returns:
            The preprocessed image.
        """
        if not self.preprocess_enabled:
            return img

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * self.preprocess_s_gain, 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * self.preprocess_v_gain, 0, 255)
        enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return enhanced

    def align_imgsz(self, size: int) -> int:
        """
        Aligns the image size to be a multiple of the model's stride.

        Args:
            size: The target image size.

        Returns:
            The aligned image size.
        """
        stride = 32
        return int(np.ceil(size / stride) * stride)

    def run_player_detection(self, img: np.ndarray, conf_thres: float) -> Tuple[List[Dict[str, Any]], Any]:
        """
        Runs player detection on a given image.

        Args:
            img: The input image.
            conf_thres: The confidence threshold for detection.

        Returns:
            A tuple containing a list of detected player targets and the raw results.
        """
        if not self.player_model:
            return [], None
        
        # For ONNX models, use CUDA only if properly supported
        if self.is_player_onnx:
            device_arg = 0 if (self.device.type == 'cuda' and self.onnx_cuda_supported) else 'cpu'
            use_half = False  # ONNX models are already optimized
        else:
            device_arg = 0 if self.device.type == 'cuda' else 'cpu'
            use_half = self.use_half
            
        imgsz_arg = self.player_onnx_input_size if self.is_player_onnx and self.player_onnx_input_size else self.config['detection']['predict_size']
        results = self.player_model(
            img,
            imgsz=imgsz_arg,
            device=device_arg,
            conf=conf_thres,
            verbose=False,
            half=use_half
        )
        return self.extract_targets(results), results

    def run_candy_detection(self, img: np.ndarray, conf_thres: float, no_detect_frames: int) -> Tuple[List[Dict[str, Any]], Any, str]:
        """
        Runs multi-scale candy detection.

        Args:
            img: The input image.
            conf_thres: The base confidence threshold.
            no_detect_frames: Number of consecutive frames with no detections.

        Returns:
            A tuple containing targets, results, and the name of the detection pass.
        """
        detection_passes: List[Dict[str, Any]] = []
        detect_cfg = self.config['detection']
        
        detection_passes.append({
            'name': 'base',
            'imgsz': self.align_imgsz(detect_cfg['predict_size']),
            'conf': conf_thres
        })

        if detect_cfg['multi_scale_enabled']:
            boost_imgsz = self.align_imgsz(int(detect_cfg['predict_size'] * detect_cfg['multi_scale_factor']))
            boost_imgsz = min(detect_cfg['heavy_scan_imgsz'], max(detect_cfg['predict_size'], boost_imgsz))
            if boost_imgsz > detect_cfg['predict_size']:
                detection_passes.append({
                    'name': 'boost',
                    'imgsz': boost_imgsz,
                    'conf': max(detect_cfg['multi_scale_conf_floor'], conf_thres + detect_cfg['multi_scale_conf_shift'])
                })

        heavy_ready = detect_cfg['multi_scale_enabled'] and no_detect_frames >= detect_cfg['heavy_scan_interval']
        if heavy_ready:
            detection_passes.append({
                'name': 'heavy',
                'imgsz': self.align_imgsz(detect_cfg['heavy_scan_imgsz']),
                'conf': detect_cfg['heavy_conf_floor']
            })
        
        # For ONNX models, use CUDA only if properly supported
        if self.is_candy_onnx:
            device_arg = 0 if (self.device.type == 'cuda' and self.onnx_cuda_supported) else 'cpu'
            use_half = False  # ONNX models are already optimized
        else:
            device_arg = 0 if self.device.type == 'cuda' else 'cpu'
            use_half = self.use_half
        
        results: Any = None
        for cfg in detection_passes:
            # If using an ONNX candy model with a fixed input size, force the imgsz
            imgsz_arg = self.candy_onnx_input_size if self.is_candy_onnx and self.candy_onnx_input_size else cfg['imgsz']

            print(f"INFO: Running detection pass '{cfg['name']}' with size={imgsz_arg}, device={device_arg}, half={use_half}")
            if self.is_candy_onnx:
                print(f"INFO: Using ONNX model with detected input size: {getattr(self, 'candy_onnx_input_size', None)}")
            
            results = self.model(
                img,
                imgsz=imgsz_arg,
                device=device_arg,
                conf=cfg['conf'],
                verbose=False,
                half=use_half,
                augment=False,
                agnostic_nms=True
            )
            
            targets = self.extract_targets(results)
            if targets:
                return targets, results, cfg['name']

        return [], results, detection_passes[-1]['name'] if detection_passes else 'base'

    def extract_targets(self, results: Any, target_cls: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Extracts target dictionaries from YOLO results.

        Args:
            results: The raw results from a YOLO model.
            target_cls: If specified, only extracts targets of this class.

        Returns:
            A list of target dictionaries.
        """
        targets: List[Dict[str, Any]] = []
        if results is None:
            return targets
            
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            for i in range(len(boxes)):
                conf: float = boxes.conf[i].item()
                cls_id: int = int(boxes.cls[i].item())

                if target_cls is None or cls_id == target_cls:
                    xyxy: np.ndarray = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = xyxy
                    targets.append({
                        'cx': (x1 + x2) / 2,
                        'cy': (y1 + y2) / 2,
                        'conf': conf,
                        'bbox': xyxy,
                        'cls': cls_id
                    })
        return targets
