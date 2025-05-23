#!/bin/bash
# run_dataset.sh

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

# Only baseline condition here; extend to weather_conditions if you like
baseline_weather_conditions=( "0 0 0" )

# Expected maximum number of images per run
EXPECTED_IMAGES=200
EXTRACT_TIMEOUT="5m"
MAX_ATTEMPTS=50

# The three towns we want to generate outputs for:
TOWNS=( Town01 Town03 Town05 )

for town in "${TOWNS[@]}"; do
  echo "=========================================="
  echo " Processing CARLA map: ${town}"
  echo "=========================================="

  for group in $(seq 1 10); do
    walker_filter="${monk_groups[$group]}"
    echo
    echo "---- Monk Skin Tone Group ${group} (filter=${walker_filter}) ----"

    for condition in "${baseline_weather_conditions[@]}"; do
      set -- $condition
      CLOUD=$1; RAIN=$2; FOG=$3

      echo
      echo "  Weather: cloud=${CLOUD}%, rain=${RAIN}%, fog=${FOG}%"

      # prefix with town so each map gets its own tree
      OUTPUT_DIR="${town}/monk_group_${group}/cloud_${CLOUD}_rain_${RAIN}_fog_${FOG}"
      mkdir -p "${OUTPUT_DIR}/RGB"

      # count already captured images
      num_images=$(find "${OUTPUT_DIR}/RGB" -type f -name "*.png" | wc -l)
      if [ "$num_images" -ge "$EXPECTED_IMAGES" ]; then
        echo "  → Skipping ${OUTPUT_DIR} (already ${num_images} images)."
        continue
      fi

      missing=$(( EXPECTED_IMAGES - num_images ))
      echo "  → Found ${num_images}, need ${missing} more."

      attempt=1
      success=0
      while [ $attempt -le $MAX_ATTEMPTS ]; do
        echo "  [${town}][group ${group}] Attempt ${attempt}/${MAX_ATTEMPTS}..."

        # launch spawn.py with --town
        python spawn.py \
          --town "${town}" \
          --number-of-vehicles 20 \
          --number-of-walkers 100 \
          --filterw "${walker_filter}" &
        SPAWN_PID=$!
        sleep 10  # allow spawns to settle

        # apply the (baseline) weather
        python weather.py --cloudiness "${CLOUD}" --rain "${RAIN}" --fog "${FOG}" &
        WEATHER_PID=$!
        sleep 5

        # re-count and run extractor
        num_images=$(find "${OUTPUT_DIR}/RGB" -type f -name "*.png" | wc -l)
        missing=$(( EXPECTED_IMAGES - num_images ))
        timeout ${EXTRACT_TIMEOUT} \
          python self_extract.py \
            --output_dir "${OUTPUT_DIR}" \
            --max_images ${missing} \
            --start_index ${num_images}
        EXIT=$?

        # tear down
        kill ${WEATHER_PID} 2>/dev/null; sleep 2
        kill ${SPAWN_PID}   2>/dev/null; sleep 5

        num_images=$(find "${OUTPUT_DIR}/RGB" -type f -name "*.png" | wc -l)
        if [ "$num_images" -ge "$EXPECTED_IMAGES" ]; then
          echo "  ✓ Captured ${num_images} images."
          success=1
          break
        else
          echo "  ✗ Only ${num_images}, retrying..."
        fi

        attempt=$(( attempt + 1 ))
      done

      if [ $success -ne 1 ]; then
        echo "  !!! Failed to reach ${EXPECTED_IMAGES} images in ${OUTPUT_DIR}"
      fi
    done
  done
done

echo
echo "All towns, all groups, baseline weather done."
