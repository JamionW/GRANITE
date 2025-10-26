# Quick validation script
sample_addresses = addresses.sample(20)

for _, addr in sample_addresses.iterrows():
    addr_times = travel_times[travel_times['origin_id'] == addr['address_id']]
    
    print(f"\nAddress {addr['address_id']}:")
    print(f"  Location: ({addr.geometry.y:.4f}, {addr.geometry.x:.4f})")
    
    for _, dest_time in addr_times.head(3).iterrows():
        straight_km = dest_time['straight_distance_km']
        network_km = dest_time['network_distance_km']
        drive_min = dest_time['drive_time']
        
        print(f"  Destination {dest_time['destination_id']}:")
        print(f"    Distance: {straight_km:.2f}km straight, {network_km:.2f}km network")
        print(f"    Drive time: {drive_min:.1f}min ({network_km/drive_min*60:.1f} km/h)")
        
        # Flag unrealistic values
        if drive_min < 2 and network_km > 1:
            print(f"    ⚠️  SUSPICIOUS: {network_km:.1f}km in {drive_min:.1f}min")