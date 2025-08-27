"use client";
import { Avatar, AvatarImage, AvatarFallback } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";

type Face = {
  id: string;
  name: string;
  image: string;
  whitelisted: boolean;
};

interface FacePreviewProps {
  faces: Face[];
}

export function FacePreview({ faces }: FacePreviewProps) {
  return (
    <Card className="p-4 bg-white dark:bg-black border border-gray-200 dark:border-gray-700">
      <div className="mb-3">
        <h4 className="font-semibold text-sm text-black dark:text-white flex items-center gap-2">
          Face Recognition Preview
        </h4>
        <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
          Shows how faces will appear during streaming
        </p>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
        {faces.map((face) => (
          <div key={face.id} className="flex flex-col items-center space-y-2">
            <div className="relative">
              <Avatar className="w-16 h-16 border-2 border-black dark:border-white shadow-lg">
                <AvatarImage
                  src={face.image}
                  alt={face.name}
                  className={face.whitelisted ? "" : "blur-md"}
                />
                <AvatarFallback
                  className={
                    face.whitelisted
                      ? "bg-gray-100 text-black dark:bg-gray-800 dark:text-white"
                      : "bg-gray-200 text-black dark:bg-gray-700 dark:text-white"
                  }
                >
                  {face.name[0]}
                </AvatarFallback>
              </Avatar>
              <div className="absolute -top-1 -right-1">
                {face.whitelisted ? (
                  <div className="bg-black dark:bg-white rounded-full p-1">
                    <span className="text-white dark:text-black text-xs">
                      ✓
                    </span>
                  </div>
                ) : (
                  <div className="bg-gray-500 rounded-full p-1">
                    <span className="text-white text-xs">✗</span>
                  </div>
                )}
              </div>
            </div>
            <div className="text-center">
              <Badge
                variant={face.whitelisted ? "default" : "destructive"}
                className={
                  face.whitelisted
                    ? "text-xs px-2 py-1 bg-black text-white dark:bg-white dark:text-black"
                    : "text-xs px-2 py-1 bg-gray-500 text-white"
                }
              >
                {face.whitelisted ? "Clear" : "Blurred"}
              </Badge>
              <p
                className="text-xs mt-1 font-medium max-w-20 truncate text-black dark:text-white"
                title={face.name}
              >
                {face.name}
              </p>
            </div>
          </div>
        ))}
      </div>
      {faces.length === 0 && (
        <div className="text-center py-8 text-gray-500 dark:text-gray-400">
          <div className="text-2xl mb-2">🛡️</div>
          <p className="text-sm">No faces to preview yet</p>
        </div>
      )}
    </Card>
  );
}
