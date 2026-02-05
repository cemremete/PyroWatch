import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';

// Fix for default markers in react-leaflet
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

function Dashboard() {
  const [fires, setFires] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdate, setLastUpdate] = useState(null);

  useEffect(() => {
    fetchFires();
    // Set up auto-refresh every 5 minutes
    const interval = setInterval(fetchFires, 300000);
    return () => clearInterval(interval);
  }, []);

  const fetchFires = async () => {
    try {
      setLoading(true);
      setError(null);
      
      console.log('Fetching fire data from PyroWatch API...');
      const response = await fetch('http://localhost:8001/api/fires/recent?hours=24&limit=100');
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      console.log(`Received ${data.length} fire records`);
      setFires(data);
      setLastUpdate(new Date());
      setError(null);
    } catch (err) {
      console.error('Error fetching fire data:', err);
      setError('Failed to load fire data. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  const getMarkerColor = (confidence) => {
    if (confidence >= 80) return '#ff0000'; // Red - High confidence
    if (confidence >= 50) return '#ff8c00'; // Orange - Medium confidence  
    return '#ffff00'; // Yellow - Low confidence
  };

  const createCustomIcon = (confidence) => {
    return L.divIcon({
      html: `<div style="background-color: ${getMarkerColor(confidence)}; width: 12px; height: 12px; border-radius: 50%; border: 2px solid white; box-shadow: 0 0 4px rgba(0,0,0,0.3);"></div>`,
      iconSize: [12, 12],
      className: 'custom-marker'
    });
  };

  const getConfidenceLabel = (confidence) => {
    if (confidence >= 80) return 'High';
    if (confidence >= 50) return 'Medium';
    return 'Low';
  };

  const refreshData = () => {
    console.log('Manual refresh triggered');
    fetchFires();
  };

  if (loading && fires.length === 0) {
    return (
      <div style={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center', 
        height: '100vh',
        backgroundColor: '#1a1a1a',
        color: 'white'
      }}>
        <div style={{ textAlign: 'center' }}>
          <h2>Loading PyroWatch...</h2>
          <p>Fetching real-time fire data from NASA satellites</p>
          <div style={{ marginTop: '20px' }}>
            <div style={{ 
              width: '40px', 
              height: '40px', 
              border: '4px solid #f3f3f3', 
              borderTop: '4px solid #ff6b6b',
              borderRadius: '50%',
              animation: 'spin 1s linear infinite'
            }}></div>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center', 
        height: '100vh',
        backgroundColor: '#1a1a1a',
        color: 'white'
      }}>
        <div style={{ textAlign: 'center' }}>
          <h2>⚠️ Error</h2>
          <p>{error}</p>
          <button 
            onClick={refreshData}
            style={{
              padding: '12px 24px',
              backgroundColor: '#ff6b6b',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer',
              fontSize: '16px',
              marginTop: '20px'
            }}
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div style={{ height: '100vh', width: '100%', position: 'relative' }}>
      <MapContainer 
        center={[39.8283, -98.5795]} // Center of USA
        zoom={4} 
        style={{ height: '100%', width: '100%' }}
      >
        <TileLayer
          url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
        />
        
        {fires.map((fire) => (
          <Marker
            key={fire.id}
            position={[fire.latitude, fire.longitude]}
            icon={createCustomIcon(fire.confidence)}
          >
            <Popup>
              <div style={{ minWidth: '220px', fontFamily: 'Arial, sans-serif' }}>
                <h4 style={{ margin: '0 0 12px 0', color: '#333', borderBottom: '2px solid #ff6b6b', paddingBottom: '8px' }}>
                  📍 Fire Detection
                </h4>
                <div style={{ fontSize: '14px', lineHeight: '1.6' }}>
                  <div><strong>Confidence:</strong> 
                    <span style={{ 
                      color: getMarkerColor(fire.confidence),
                      fontWeight: 'bold',
                      marginLeft: '8px'
                    }}>
                      {fire.confidence.toFixed(1)}% ({getConfidenceLabel(fire.confidence)})
                    </span>
                  </div>
                  <div><strong>Brightness:</strong> {fire.brightness}K</div>
                  <div><strong>Satellite:</strong> {fire.satellite}</div>
                  <div><strong>Date:</strong> {fire.acq_date} {fire.acq_time}</div>
                  {fire.frp && (
                    <div><strong>Fire Radiative Power:</strong> {fire.frp} MW</div>
                  )}
                  {fire.daynight && (
                    <div><strong>Day/Night:</strong> {fire.daynight === 'D' ? 'Day' : 'Night'}</div>
                  )}
                </div>
              </div>
            </Popup>
          </Marker>
        ))}
      </MapContainer>
      
      {/* Status Bar */}
      <div style={{
        position: 'absolute',
        top: '15px',
        right: '15px',
        backgroundColor: 'rgba(0,0,0,0.85)',
        color: 'white',
        padding: '12px 18px',
        borderRadius: '8px',
        fontSize: '14px',
        zIndex: 1000,
        backdropFilter: 'blur(10px)',
        border: '1px solid rgba(255,255,255,0.1)'
      }}>
        <div style={{ fontWeight: 'bold', marginBottom: '8px' }}>
          🌡️ PyroWatch
        </div>
        <div>{fires.length} active fires detected</div>
        <div style={{ fontSize: '12px', marginTop: '6px', opacity: 0.8 }}>
          Last 24 hours
        </div>
        {lastUpdate && (
          <div style={{ fontSize: '11px', marginTop: '8px', opacity: 0.6 }}>
            Updated: {lastUpdate.toLocaleTimeString()}
          </div>
        )}
      </div>

      {/* Legend */}
      <div style={{
        position: 'absolute',
        bottom: '25px',
        right: '15px',
        backgroundColor: 'rgba(0,0,0,0.85)',
        color: 'white',
        padding: '12px',
        borderRadius: '8px',
        fontSize: '12px',
        zIndex: 1000,
        backdropFilter: 'blur(10px)',
        border: '1px solid rgba(255,255,255,0.1)'
      }}>
        <div style={{ marginBottom: '12px', fontWeight: 'bold' }}>
          🔥 Confidence Level
        </div>
        <div style={{ display: 'flex', alignItems: 'center', marginBottom: '6px' }}>
          <div style={{ 
            width: '12px', 
            height: '12px', 
            backgroundColor: '#ff0000', 
            borderRadius: '50%',
            marginRight: '10px'
          }}></div>
          <span>High (≥80%)</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', marginBottom: '6px' }}>
          <div style={{ 
            width: '12px', 
            height: '12px', 
            backgroundColor: '#ff8c00', 
            borderRadius: '50%',
            marginRight: '10px'
          }}></div>
          <span>Medium (50-80%)</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <div style={{ 
            width: '12px', 
            height: '12px', 
            backgroundColor: '#ffff00', 
            borderRadius: '50%',
            marginRight: '10px'
          }}></div>
          <span>Low (&lt;50%)</span>
        </div>
      </div>

      {/* Refresh Button */}
      <button
        onClick={refreshData}
        style={{
          position: 'absolute',
          top: '15px',
          left: '15px',
          backgroundColor: 'rgba(0,0,0,0.85)',
          color: 'white',
          border: '1px solid rgba(255,255,255,0.2)',
          borderRadius: '8px',
          padding: '10px 16px',
          cursor: 'pointer',
          fontSize: '14px',
          zIndex: 1000,
          backdropFilter: 'blur(10px)'
        }}
        title="Refresh fire data"
      >
        🔄 Refresh
      </button>

      {/* Loading indicator for background refresh */}
      {loading && fires.length > 0 && (
        <div style={{
          position: 'absolute',
          top: '70px',
          left: '15px',
          backgroundColor: 'rgba(255,107,107,0.9)',
          color: 'white',
          padding: '8px 12px',
          borderRadius: '6px',
          fontSize: '12px',
          zIndex: 1000
        }}>
          Updating...
        </div>
      )}

      {/* Add CSS for spinner animation */}
      <style jsx>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}

export default Dashboard;
