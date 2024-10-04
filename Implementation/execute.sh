echo "Updating package lists..."
sudo apt-get update

echo "Updating package lists..."
pip install -r requirements.txt

if [ ! -d "reference_images" ]; then
    echo "Error: 'reference_images' folder not found!"
    exit 1
fi

if [ ! -d "two_players_bot" ]; then
    echo "Error: 'two_players_bot' folder not found!"
    exit 1
fi

if [ ! -d "two_players_top" ]; then
    echo "Error: 'two_players_top' folder not found!"
    exit 1
fi

python3 main.py

if [ -d "output" ]; then
    echo "Output folder successfully created!"
    echo "Check the 'output' directory for segregated player folders."
else
    echo "Error: Output folder was not created."
    exit 1
fi
