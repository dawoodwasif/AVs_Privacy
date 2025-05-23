#!/bin/bash
# run_dataset.sh

# Define CARLA towns
towns=("Town01" "Town03" "Town05")

# Define monk skin tone groups as an associative array.
declare -A monk_groups=(
  [1]="walker.pedestrian.0001"
  [2]="walker.pedestrian.0016"
  [3]="walker.pedestrian.0034"
  [4]="walker.pedestrian.0027"
  [5]="walker.pedestrian.0021"
  [6]="walker.pedestrian.0023"
  [7]="walker.pedestrian.0024"
  [8]="walker.pedestrian.0026"
  [9]="walker.pedestrian.0041"
  [10]="walker.pedestrian.0042"
)

# Define all 13 weather conditions (cloudiness rain fog)
weather_conditions=(
    "0 0 0"
    "25 0 0"
    "50 0 0"
    "75 0 0"
    "100 0 0"
    "0 25 0"
    "0 50 0"
    "0 75 0"
    "0 100 0"
    "0 0 25"
    "0 0 50"
    "0 0 75"
    "0 0 100"
)

# Expected maximum number of images per run
EXPECTED_IMAGES=200

# Timeout for self_extract.py
EXTRACT_TIMEOUT="5m"

# Maximum number of attempts per subfolder
MAX_ATTEMPTS=50

for town in "${towns[@]}"; do
    echo "=========================================="
    echo "Processing CARLA Town: ${town}"
    echo "=========================================="
    
    # create top‐level directory for this town
    mkdir -p "carla_dataset/${town}"

    for group in $(seq 1 10); do
        walker_filter="${monk_groups[$group]}"
        echo "  --------------------------------------"
        echo "  Monk Skin Tone Group ${group}: ${walker_filter}"
        echo "  --------------------------------------"

        for condition in "${weather_conditions[@]}"; do
            # parse weather triple
            read -r CLOUDINESS RAIN FOG <<< "${condition}"

            echo "    Weather: cloud=${CLOUDINESS}%, rain=${RAIN}%, fog=${FOG}%"

            # set the output directory
            OUTPUT_DIR="carla_dataset/${town}/monk_group_${group}/cloud_${CLOUDINESS}_rain_${RAIN}_fog_${FOG}"
            mkdir -p "${OUTPUT_DIR}/RGB"

            # count existing images
            num_images=$(find "${OUTPUT_DIR}/RGB" -type f -name "*.png" | wc -l)
            if [ "$num_images" -ge "$EXPECTED_IMAGES" ]; then
                echo "    Skipping ${OUTPUT_DIR} (already ${num_images} images)."
                continue
            fi

            missing=$(( EXPECTED_IMAGES - num_images ))
            echo "    Found ${num_images}, need ${missing} more."

            attempt=1
            success=0
            while [ $attempt -le $MAX_ATTEMPTS ]; do
                echo "      Attempt ${attempt}..."

                # spawn.py with town, vehicle & walker counts, filter
                python spawn.py \
                  --town "${town}" \
                  --number-of-vehicles 20 \
                  --number-of-walkers 100 \
                  --filterw "${walker_filter}" &
                SPAWN_PID=$!
                sleep 10  # let actors spawn

                # set the weather
                python weather.py \
                  --cloudiness "${CLOUDINESS}" \
                  --rain "${RAIN}" \
                  --fog "${FOG}" &
                WEATHER_PID=$!
                sleep 5   # let weather apply

                # recalc how many images we already have
                num_images=$(find "${OUTPUT_DIR}/RGB" -type f -name "*.png" | wc -l)
                missing=$(( EXPECTED_IMAGES - num_images ))
                echo "      Capturing ${missing} images..."

                timeout ${EXTRACT_TIMEOUT} python self_extract.py \
                  --output_dir "${OUTPUT_DIR}" \
                  --max_images ${missing} \
                  --start_index ${num_images}
                EXIT_STATUS=$?
                if [ $EXIT_STATUS -eq 124 ]; then
                    echo "      self_extract.py timed out."
                elif [ $EXIT_STATUS -ne 0 ]; then
                    echo "      self_extract.py error (status ${EXIT_STATUS})."
                else
                    echo "      Extraction completed."
                fi

                # tear down
                kill ${WEATHER_PID} 2>/dev/null; sleep 2
                kill ${SPAWN_PID} 2>/dev/null;   sleep 5

                # check if done
                num_images=$(find "${OUTPUT_DIR}/RGB" -type f -name "*.png" | wc -l)
                if [ "$num_images" -ge "$EXPECTED_IMAGES" ]; then
                    echo "    ✔ Captured ${num_images} images."
                    success=1
                    break
                else
                    missing=$(( EXPECTED_IMAGES - num_images ))
                    echo "    Still missing ${missing}, retrying..."
                fi

                attempt=$((attempt+1))
            done

            if [ $success -ne 1 ]; then
                echo "    ✖ Failed after ${MAX_ATTEMPTS} attempts."
            fi

        done
    done
done

echo "All dataset runs complete."
