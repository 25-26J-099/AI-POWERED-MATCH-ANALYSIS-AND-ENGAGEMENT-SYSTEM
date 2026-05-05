import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { ThemeProvider } from './hooks/useTheme';
import Navbar from './components/Navbar';
import Landing from './pages/Landing';
import Upload from './pages/Upload';
import Processing from './pages/Processing';
import MatchDashboard from './pages/MatchDashboard';
import PlayerAnalysis from './pages/PlayerAnalysis';
import PlayerComparison from './pages/PlayerComparison';
import StyleMap from './pages/StyleMap';
import AiAnalysis from './pages/AiAnalysis';
import Commentary from './pages/Commentary';
import DecisionQuality from './pages/DecisionQuality';
import LineupSetup from './pages/LineupSetup';

function App() {
  return (
    <ThemeProvider>
      <BrowserRouter>
        <Navbar />
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/upload" element={<Upload />} />
          <Route path="/lineup/:id" element={<LineupSetup />} />
          <Route path="/processing/:id" element={<Processing />} />
          <Route path="/match/:id" element={<MatchDashboard />} />
          <Route path="/match/:id/player/:playerId" element={<PlayerAnalysis />} />
          <Route path="/match/:id/compare" element={<PlayerComparison />} />
          <Route path="/match/:id/styles" element={<StyleMap />} />
          <Route path="/match/:id/analysis" element={<AiAnalysis />} />
          <Route path="/match/:id/commentary" element={<Commentary />} />
          <Route path="/match/:id/decision-quality" element={<DecisionQuality />} />
        </Routes>
      </BrowserRouter>
    </ThemeProvider>
  );
}

export default App;
