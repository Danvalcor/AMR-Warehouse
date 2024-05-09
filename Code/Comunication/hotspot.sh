#!/bin/bash

# Cambiar al directorio ~/AccessPopup
cd ~/AccessPopup || { echo "Error: No se pudo cambiar al directorio ~/AccessPopup"; exit 1; }

# Ejecutar el comando accesspopup con la opción -a
./accesspopup -a || { echo "Error: No se pudo ejecutar el comando accesspopup"; exit 1; }

