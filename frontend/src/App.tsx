import UploadForm from './components/UploadForm';
import './App.css'
import { ChakraProvider } from '@chakra-ui/react';

function App() {
  return (
    <>
      <div>
        <h1>Welcome to SydeKick AI!</h1>
      </div>
      <ChakraProvider>
        <UploadForm/>
      </ChakraProvider>
    </>
  )
}

export default App
