def read_all_fit_files(folder_path):
    '''
    Read all fit files in the folder
    '''
    record_dict = defaultdict(dict)
    for filename in os.listdir(folder_path):
        if filename.endswith('.fit'):
            file_path = os.path.join(folder_path, filename)
            try:
                fitfile = FitFile(file_path, check_crc=False)
                heart_rate_list = []
                speed_list = []
                distance_list = []
                timestamp_list = []
                date_str = None
                for record in fitfile.get_messages('record'):
                    record_data = {d.name: d.value for d in record}
                    if record_data.get('activity_type') == 'running' and 'heart_rate' in record_data and 'speed' in record_data and 'timestamp' in record_data and 'distance' in record_data:
                        try:
                            timestamp = record_data['timestamp']
                            heart_rate = record_data['heart_rate']
                            speed = record_data['speed']
                            distance = record_data['distance']
                            if heart_rate is None or speed is None or distance is None:
                                continue
                            # Store daily data
                            timestamp_list.append(timestamp)
                            heart_rate_list.append(heart_rate)
                            speed_list.append(speed)
                            distance_list.append(distance)
                            date_str = timestamp.date().isoformat()
                        except Exception as e:
                            pass
                record_dict[date_str] = {"heart_rate": heart_rate_list, "speed": speed_list, "distance": distance_list, "timestamp": timestamp_list, "date": date_str,distance:distance_list[-1]}
            except Exception as e:
                pass
    return record_dict

