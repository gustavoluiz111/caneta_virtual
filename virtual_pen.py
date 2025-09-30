import cv2
import numpy as np
import mediapipe as mp

# --- Configurações de Cores e Tamanhos ---
# Cores no formato BGR (Blue, Green, Red)
COLORS = {
    "BLUE": (255, 0, 0),
    "GREEN": (0, 255, 0),
    "RED": (0, 0, 255),
    "ERASER": (0, 0, 0) # Preto, para "apagar" no canvas preto
}
DRAW_THICKNESS = 8
ERASER_THICKNESS = 50 

# --- Inicialização do MediaPipe e Câmera ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Largura (W)
cap.set(4, 720)   # Altura (H)

# Cria o "quadro branco" (canvas) onde o desenho será armazenado
canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

# Variáveis de controle
draw_color = COLORS["GREEN"] # Cor inicial
prev_x, prev_y = 0, 0        # Posição anterior do dedo

# --- Função de Reconhecimento de Gestos ---
def get_drawing_mode(landmarks):
    """
    Identifica o modo de desenho baseado nos dedos esticados.
    Retorna o modo ('DRAW', 'ERASE', 'IDLE').
    """
    # Pontas dos dedos: Indicador=8, Médio=12, Anelar=16, Mindinho=20
    finger_tips = [8, 12, 16, 20]
    # Articulações do meio: Abaixo do Indicador=6, Médio=10, etc.
    pips = [6, 10, 14, 18]       
    
    fingers_up = []
    # Checa se os dedos estão esticados (ponta acima da articulação do meio)
    for tip_index, pip_index in zip(finger_tips, pips):
        is_up = landmarks.landmark[tip_index].y < landmarks.landmark[pip_index].y
        fingers_up.append(is_up)

    # Lógica dos Modos:
    # 1. DRAW: Apenas o Indicador está esticado
    if fingers_up[0] and not fingers_up[1] and not fingers_up[2] and not fingers_up[3]:
        return 'DRAW'
    
    # 2. ERASE: Indicador e Médio estão esticados (gesto de tesoura)
    elif fingers_up[0] and fingers_up[1] and not fingers_up[2] and not fingers_up[3]:
        return 'ERASE'
        
    # 3. IDLE: Qualquer outro gesto (ou mão fechada)
    else:
        return 'IDLE'

# --- Loop Principal ---
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Erro ao ler o frame da câmera.")
        break

    # 1. Pré-processamento
    frame = cv2.flip(frame, 1) # Experiência de espelho
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    h, w, _ = frame.shape

    # 2. Desenho da Interface (UI) - Zonas de Cor e Indicador
    # Zonas de Seleção (Barra superior: 0 a 100 pixels de altura)
    cv2.rectangle(frame, (200, 0), (400, 100), COLORS["BLUE"], -1)
    cv2.rectangle(frame, (500, 0), (700, 100), COLORS["GREEN"], -1)
    cv2.rectangle(frame, (800, 0), (1000, 100), COLORS["RED"], -1)
    # Indicador de cor atual
    cv2.circle(frame, (1150, 50), 30, draw_color, -1)
    cv2.putText(frame, "Cor", (1080, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
            # Posição da ponta do dedo indicador (landmark 8)
            index_tip_lm = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            current_x = int(index_tip_lm.x * w)
            current_y = int(index_tip_lm.y * h)
            
            # 3. Lógica de Seleção de Cor (Se o dedo estiver no topo)
            if current_y < 100:
                prev_x, prev_y = 0, 0 # Reseta o traço para não desenhar
                
                if 200 < current_x < 400:
                    draw_color = COLORS["BLUE"]
                elif 500 < current_x < 700:
                    draw_color = COLORS["GREEN"]
                elif 800 < current_x < 1000:
                    draw_color = COLORS["RED"]
                
                cv2.putText(frame, "SELECIONANDO COR", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # 4. Lógica de Desenho/Borracha (Se o dedo estiver abaixo da área de seleção)
            else:
                mode = get_drawing_mode(hand_landmarks)

                if mode == 'DRAW':
                    # Desenha o círculo de ponta no frame (para feedback visual)
                    cv2.circle(frame, (current_x, current_y), DRAW_THICKNESS, draw_color, cv2.FILLED)
                    cv2.putText(frame, "DESENHAR", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, draw_color, 2)
                    
                    if prev_x == 0 and prev_y == 0:
                        prev_x, prev_y = current_x, current_y
                    
                    # Desenha a linha no canvas (o "quadro branco" persistente)
                    cv2.line(canvas, (prev_x, prev_y), (current_x, current_y), draw_color, DRAW_THICKNESS)
                    prev_x, prev_y = current_x, current_y

                elif mode == 'ERASE':
                    # Desenha o círculo de borracha no frame (visual)
                    cv2.circle(frame, (current_x, current_y), ERASER_THICKNESS, (255, 255, 255), cv2.FILLED) 
                    cv2.putText(frame, "BORRACHA", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    if prev_x == 0 and prev_y == 0:
                        prev_x, prev_y = current_x, current_y

                    # Borracha é desenhar com a cor PRETA no canvas
                    cv2.line(canvas, (prev_x, prev_y), (current_x, current_y), COLORS["ERASER"], ERASER_THICKNESS)
                    prev_x, prev_y = current_x, current_y
                
                else: # IDLE (ocioso/mover o cursor)
                    prev_x, prev_y = 0, 0
                    cv2.putText(frame, "MOVER/OCIOSO", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
            # Desenha os landmarks da mão (opcional, para visualização)
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # 5. Mesclagem do Canvas e Frame (Crucial para o Desenho Persistente)
    img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    # Cria uma máscara inversa (tudo que não é preto vira branco)
    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
    
    # Isola o desenho
    frame = cv2.bitwise_and(frame, img_inv)
    # Adiciona o desenho ao frame da câmera
    frame = cv2.bitwise_or(frame, canvas)   
    
    # 6. Exibição e Controles
    cv2.imshow("Caneta Virtual AI (Pressione 'q' para sair)", frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    # Limpar o canvas (Tecla 'c')
    if key == ord('c'):
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Sair do programa (Tecla 'q')
    if key == ord('q'):
        break

# Encerramento
cap.release()
cv2.destroyAllWindows()
