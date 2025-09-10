import os
import shutil
import cv2
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter

def create_dataset_directories(base_dir='quinoa_dataset'):
    """
    Crea la estructura de directorios para el dataset.
    """
    print("Creando directorios del dataset...")
    # Directorio principal del dataset
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir)

    # Directorios para entrenamiento, validación y prueba
    for subset in ['train', 'val', 'test']:
        subset_path = os.path.join(base_dir, subset)
        os.makedirs(subset_path, exist_ok=True)
        # Clases de enfermedades
        for disease in ['Mancha_Foliar', 'Mancha_Bacteria', 'Mildiu', 'Sana']:
            os.makedirs(os.path.join(subset_path, disease), exist_ok=True)
    print("Directorios creados exitosamente.")

def augment_image_with_deformations(image, num_augmentations=10):
    """
    Aplica una variedad de técnicas de aumento de datos a una imagen
    para generar un número específico de imágenes aumentadas.
    """
    augmented_images = []
    
    for i in range(num_augmentations):
        # Clonar la imagen original para aplicar transformaciones
        current_image = image.copy()
        
        # Voltear horizontalmente de forma aleatoria (50% de probabilidad)
        if np.random.rand() > 0.5:
            current_image = cv2.flip(current_image, 1)
        
        # Rotar aleatoriamente
        angle = np.random.randint(-15, 15)
        (h, w) = current_image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        current_image = cv2.warpAffine(current_image, M, (w, h))
        
        # Ajustar brillo y contraste (50% de probabilidad)
        if np.random.rand() > 0.5:
            alpha = np.random.uniform(low=0.8, high=1.2)  # Contraste
            beta = np.random.uniform(low=-20, high=20)   # Brillo
            current_image = cv2.convertScaleAbs(current_image, alpha=alpha, beta=beta)
        
        # Aplicar deformación elástica a algunas imágenes (60% de probabilidad)
        if np.random.rand() > 0.4:
            alpha_deform = np.random.uniform(low=10, high=30)
            sigma_deform = np.random.uniform(low=3, high=5)
            current_image = deform_image(current_image, alpha_deform, sigma_deform)

        augmented_images.append(current_image)
        
    return augmented_images

def deform_image(image, alpha, sigma, random_state=None):
    """
    Aplica una deformación elástica a la imagen.
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    # Se corrige el orden de los argumentos de meshgrid para evitar el error de broadcast
    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

    deformed_image = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
    
    return deformed_image


def process_and_save_dataset(input_images_dir, output_base_dir='quinoa_dataset', augment_factor=10):
    """
    Procesa las imágenes originales, las aumenta y las guarda en la estructura del dataset.
    """
    print("Procesando y guardando imágenes...")
    
    # Obtener las clases (nombres de las subcarpetas)
    classes = os.listdir(input_images_dir)
    
    # Recorre cada clase (carpeta de enfermedad)
    for class_name in classes:
        class_path = os.path.join(input_images_dir, class_name)
        if not os.path.isdir(class_path):
            continue # Salta si no es un directorio

        # Obtén la lista de imágenes dentro de la carpeta de la clase
        image_files = os.listdir(class_path)
        np.random.shuffle(image_files)

        # Dividir los datos (se hace para cada clase)
        train_split = 0.7
        val_split = 0.2
        train_count = int(len(image_files) * train_split)
        val_count = int(len(image_files) * val_split)

        train_images = image_files[:train_count]
        val_images = image_files[train_count:train_count + val_count]
        test_images = image_files[train_count + val_count:]

        datasets = {
            'train': train_images,
            'val': val_images,
            'test': test_images
        }

        for subset_name, images_list in datasets.items():
            for filename in images_list:
                image_path = os.path.join(class_path, filename)
                original_image = cv2.imread(image_path)
                
                if original_image is None:
                    print(f"Advertencia: No se pudo cargar la imagen {image_path}. Saltando.")
                    continue
                
                # Generar el número especificado de imágenes aumentadas
                augmented_images = augment_image_with_deformations(original_image, num_augmentations=augment_factor)

                # Guardar las imágenes procesadas
                for i, aug_img in enumerate(augmented_images):
                    output_path = os.path.join(
                        output_base_dir,
                        subset_name,
                        class_name,
                        f'{os.path.splitext(filename)[0]}_aug_{i}.jpg'
                    )
                    cv2.imwrite(output_path, aug_img)
    
    print("Procesamiento de imágenes completado.")

if __name__ == '__main__':
    # Asegúrate de que esta ruta sea la carpeta principal que contiene las subcarpetas de las clases
    input_images_directory = 'hojas_originales'
    
    create_dataset_directories()
    
    process_and_save_dataset(input_images_directory)
