import UploadForm from './components/UploadForm';
import './App.css'
import { ChakraProvider } from '@chakra-ui/react';

function App() {
  return (
    <>
      <div>
        <h1>Welcome to Galpao Da Luta!</h1>
      </div>
      <ChakraProvider>
        <UploadForm/>
      </ChakraProvider>
    </>
  )
}

export default App
