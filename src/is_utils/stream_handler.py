import numpy as np
import cv2
import threading
import time
from is_wire.core import Subscription
from is_msgs.image_pb2 import ObjectAnnotations, Image
from is_utils.streamChannel import StreamChannel 

class StreamHandler:
    """
    Gerencia a comunicação com o broker usando Threads para não bloquear o loop principal.
    """
    def __init__(self, broker_uri, num_cameras, num_keypoints, all_calibs_parameters):
   
        self.broker_uri = broker_uri
        self.skeleton_channels = []
        self._subscriptions = [] 
        self.all_calibs_parameters = all_calibs_parameters

        self.num_cameras = num_cameras
        self.num_keypoints = num_keypoints

        # Buffer para armazenar as últimas mensagens recebidas
        # Estrutura: { cam_index (0..N): mensagem_objeto }
        self._latest_skeletons = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        print(f"Conectando ao broker em {self.broker_uri}...")
        
        # Configurar canais e iniciar threads de leitura
        for i in range(1, self.num_cameras + 1):
            sk_channel = StreamChannel(self.broker_uri)
            self.skeleton_channels.append(sk_channel)
            
            sk_sub = Subscription(channel=sk_channel)
            sk_sub.subscribe(topic=f'SkeletonDetector.{i}.Detection')
            self._subscriptions.append(sk_sub)
            
            # Inicia uma thread para cada câmera
            t = threading.Thread(target=self._consume_loop, args=(sk_channel, i-1), daemon=True)
            t.start()
        
    def _consume_loop(self, channel, cam_index):
        """Loop rodando em thread separada para consumir mensagens sem parar."""
        while not self._stop_event.is_set():
            try:
                # Usa consume normal com timeout curto
                msg = channel.consume(timeout=0.05)
                with self._lock:
                    self._latest_skeletons[cam_index] = msg
            except Exception:
                # Timeout ou erro de socket, apenas continua tentando
                continue

    def get_latest_messages(self):
        """
        Retorna instantaneamente as últimas mensagens armazenadas na memória.
        """
        with self._lock:
            # Verifica se temos mensagens de todas as câmeras
            if len(self._latest_skeletons) < self.num_cameras:
                return None
            
            # Cria a lista ordenada de mensagens
            sk_msgs = [self._latest_skeletons.get(i) for i in range(self.num_cameras)]
            
            # Validação extra para garantir que ninguém é None
            if any(m is None or isinstance(m, bool) for m in sk_msgs):
                return None

            messages = {
                'skeletons': sk_msgs
            }
            return messages

    def prepare_input_data(self, messages: dict, calib_files: list):
        # Desempacota as anotações dos esqueletos
        # print('Messages: ', messages)
        skeleton_annotations = []
        for cam_index, msg in enumerate(messages['skeletons']):
            annotations = msg.unpack(ObjectAnnotations) 
            cam_params = self.all_calibs_parameters[cam_index]
            K = cam_params['K']
            nK = cam_params['nK']
            dist_coeffs = cam_params['dist_coeffs']
            for obj in annotations.objects:
                raw_kps = np.array([[kp.position.x, kp.position.y] for kp in obj.keypoints], dtype=np.float32) # (N, 2)

                if len(raw_kps) == 0:
                    continue

                pts_undistorted = cv2.undistortPoints(raw_kps.reshape(-1, 1, 2), K, dist_coeffs, P=nK).reshape(-1, 2)

                for kp, (ux, uy) in zip(obj.keypoints, pts_undistorted):
                    kp.position.x = float(ux)
                    kp.position.y = float(uy)

                if len(obj.region.vertices) >= 2:
                    v1, v2 = obj.region.vertices[0], obj.region.vertices[1]
                    corners = np.array([[v1.x, v1.y], [v2.x, v2.y]], dtype=np.float32).reshape(-1, 1, 2)
                    undistorted_corners = cv2.undistortPoints(corners, K, dist_coeffs, P=nK).reshape(-1, 2)
                    
                    obj.region.vertices[0].x, obj.region.vertices[0].y = (float(undistorted_corners[0, 0]), float(undistorted_corners[0, 1]))
                    obj.region.vertices[1].x, obj.region.vertices[1].y = (float(undistorted_corners[1, 0]), float(undistorted_corners[1, 1]))

            skeleton_annotations.append(annotations)
        return skeleton_annotations
        
    def stop(self):
        self._stop_event.set()
