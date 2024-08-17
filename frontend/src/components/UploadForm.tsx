import React, { useState } from 'react';
import {
  Box,
  Button,
  Input,
  Stack,
  Text,
  useToast,
} from '@chakra-ui/react';

const UploadForm = () => {
  const [userName, setUserName] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const toast = useToast();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!file || !userName) {
      toast({
        title: "Error",
        description: "Please provide a user name and select a file.",
        status: "error",
        duration: 3000,
        isClosable: true,
      });
      return;
    }

    const formData = new FormData();
    formData.append('user_name', userName);
    formData.append('file_name', file.name);
    formData.append('video', file);

    try {
      const response = await fetch('http://localhost:4001/upload', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        toast({
          title: "Upload successful",
          description: "Your video has been uploaded.",
          status: "success",
          duration: 3000,
          isClosable: true,
        });
      } else {
        toast({
          title: "Upload failed",
          description: "There was a problem uploading your video.",
          status: "error",
          duration: 3000,
          isClosable: true,
        });
      }
    } catch (error) {
      toast({
        title: "Network error",
        description: "Failed to connect to the server.",
        status: "error",
        duration: 3000,
        isClosable: true,
      });
    }
  };

  return (
    <Box p={4}>
      <form onSubmit={handleSubmit}>
        <Stack spacing={4}>
          <Text fontSize="lg" fontWeight="bold">Upload Your Video</Text>
          <Input
            placeholder="Enter your name"
            value={userName}
            onChange={(e) => setUserName(e.target.value)}
            required
          />
          <Input
            type="file"
            accept="video/mp4"
            onChange={(e) => setFile(e.target.files ? e.target.files[0] : null)}
            required
          />
          <Button type="submit" colorScheme="blue">Upload</Button>
        </Stack>
      </form>
    </Box>
  );
};

export default UploadForm;
